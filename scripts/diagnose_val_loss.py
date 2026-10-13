"""Break val/loss down by ordinal threshold, lead time and sample, next to the
probabilistic scores, for several checkpoints of one run -- to see where a
rising val/loss comes from while val/crps and val/csi do not move with it.

`OrdinalConsistentLoss` is a *conditional* BCE: channel k is the logit of
P(>= edge_k | >= edge_{k-1}) and is scored only on pixels whose target
reaches edge_{k-1}. Each sample averages every channel over its own
conditional pixels (lead-weighted), then averages the 16 channels with equal
weight -- so a channel scored on a handful of >= 55 dBZ pixels counts as much
as the >= 5 dBZ channel scored on millions. CRPS / Brier / reliability score
the *unconditional* exceedance probability (the cumprod of the conditionals),
where a conditional error is scaled down by how rare its condition is.

Per checkpoint, on val and on a fixed random train subset (eval mode, no
jitter), this writes:

  * per_threshold.csv -- (ckpt, split, threshold, lead_min, metric, value):
      cond_bce      conditional BCE, pooled over pixels, unweighted (what the
                    loss scores per channel)
      cond_n        number of conditional pixels behind cond_bce
      loss_contrib  this channel's share of the loss as logged (per-sample
                    lead-weighted mean, divided by the sample's channel count,
                    averaged over samples); lead_min = "all", sums to the loss
      uncond_bce    BCE of the unconditional exceedance probability
      event_rate    observed P(>= threshold)
      mean_prob     mean forecast P(>= threshold)
      brier         Brier score of P(>= threshold)
      crps_contrib  mm/h bucket gap x brier: CRPS splits exactly into these
      ece           count-weighted |forecast - observed| over 10 prob bins
  * per_sample.csv -- (ckpt, split, sample, init_time, loss, n_ge<edge>): each
    sample's loss as logged and its pixel counts at the top thresholds, to
    check whether a few samples carry a loss change.
  * figures: cond_bce.png, loss_contrib.png, probabilistic.png,
    cond_bce_lead_diff.png, sample_loss_change.png

Every checkpoint is scored on the same batches in one pass. The datamodule is
rebuilt from the first checkpoint's saved hyperparameters, so the split (CV
fold, manifest, low-echo filter) and the input downsampling match the run --
run this from the branch the run was trained on.

Usage (GPU node; add --limit-batches 10 --train-samples 64 for a quick check):
    python scripts/diagnose_val_loss.py \\
        --ckpt-dir runs/rainpro8_2021_v2_f0/checkpoints \\
        --out-dir figures/loss_diag/v2_f0
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainpro.data.rainpro8_datamodule import RainPro8DataModule  # noqa: E402
from rainpro.data.rainpro8_dataset import RainPro8Dataset  # noqa: E402
from rainpro.loss.ordinal_consistent import Bucketize, taiwan_dbz_buckets  # noqa: E402
from rainpro.metrics.probabilistic import (  # noqa: E402
    CRPS,
    BrierScore,
    ReliabilityAccumulator,
    _bucket_edges_dbz,
    _bucket_gaps_mmh,
)
from rainpro.modules.rainpro8 import RainPro8Module  # noqa: E402
from rainpro.modules.utils import EvalOutputs, EvalRequest  # noqa: E402

CKPT_NAMES = ("best_loss", "best_crps", "best_csi", "last")
CKPT_COLORS = {"best_loss": "#2a78d6", "best_crps": "#1baf7a", "best_csi": "#eb6834", "last": "#6b6b6b"}
EDGES = [b.min for b in taiwan_dbz_buckets()]
TOP_EDGES = (46.0, 52.0, 55.0, 60.0)  # per_sample.csv pixel counts
SURFACE = "#fcfcfb"


class CaptureLogits(torch.nn.Module):
    """Wraps the network's criterion to keep the raw logits it is called with."""

    def __init__(self, criterion: torch.nn.Module):
        super().__init__()
        self.criterion = criterion
        self.logits: torch.Tensor | None = None

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.logits = outputs
        return self.criterion(outputs, target)


class ThresholdStats:
    """Accumulates every per-threshold quantity for one (checkpoint, split)."""

    def __init__(self, num_leads: int, lead_weights: torch.Tensor, device: str):
        c = len(EDGES)

        def zeros():
            return torch.zeros(c, num_leads, dtype=torch.float64, device=device)

        self.cond_sum, self.cond_n, self.uncond_sum, self.event_n = zeros(), zeros(), zeros(), zeros()
        self.lead_w = lead_weights.to(device).view(1, -1, 1, 1, 1)
        self.k = torch.arange(c, device=device).view(1, 1, c, 1, 1)
        self.bucketize = Bucketize(taiwan_dbz_buckets()).to(device)
        self.crps = CRPS(num_leads).to(device)
        self.brier = BrierScore(num_leads).to(device)
        self.reliability = ReliabilityAccumulator(num_leads).to(device)
        self.sample_loss: list[torch.Tensor] = []  # (B, C) per-sample channel terms as the loss sees them
        self.sample_counts: list[torch.Tensor] = []  # (B, len(TOP_EDGES))

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(target)  # (B, T, 1, H, W)
        bucket = self.bucketize(target)
        y = (self.k <= bucket).to(logits.dtype)  # target >= edge_k
        cond = ((bucket >= self.k - 1) & valid).to(logits.dtype)  # target >= edge_{k-1}
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

        def by_threshold_lead(x):  # (B, T, C, H, W) -> (C, T)
            return x.sum(dim=(0, 3, 4)).transpose(0, 1).double()

        self.cond_sum += by_threshold_lead(bce * cond)
        self.cond_n += by_threshold_lead(cond)

        exceed = (1 - probs).clamp(1e-7, 1 - 1e-7)  # P(>= edge_k)
        validf = valid.to(logits.dtype).expand_as(exceed)
        self.uncond_sum += by_threshold_lead(-(y * exceed.log() + (1 - y) * (1 - exceed).log()) * validf)
        self.event_n += by_threshold_lead(y * validf)

        eo = EvalOutputs(forecast=None, target=target, probs=probs)
        for metric in (self.crps, self.brier, self.reliability):
            metric.update(eo)

        # Same reduction as OrdinalConsistentLoss: lead-weighted mean over each
        # sample's conditional pixels, per channel (NaN where it has none).
        n = cond.sum(dim=(1, 3, 4))
        per_channel = (bce * self.lead_w * cond).sum(dim=(1, 3, 4)) / n
        per_channel = per_channel.masked_fill(n == 0, float("nan"))
        self.sample_loss.append(per_channel.cpu())
        tgt = torch.nan_to_num(target, nan=-1.0)
        self.sample_counts.append(
            torch.stack([(tgt >= e).sum(dim=(1, 2, 3, 4)) for e in TOP_EDGES], dim=1).cpu()
        )
        return per_channel.nanmean(dim=1).mean()

    def rows(self, ckpt: str, split: str, leads: list[int]) -> list[dict]:
        out = []

        def add(metric, table):  # (C, T) -> one row per threshold and lead
            table = table.detach().double().cpu()
            for ci, edge in enumerate(EDGES):
                for li, lead_min in enumerate(leads):
                    out.append(dict(ckpt=ckpt, split=split, threshold=edge, lead_min=lead_min,
                                    metric=metric, value=float(table[ci, li])))

        valid_n = self.crps.count.double().clamp(min=1)  # valid pixels per lead
        add("cond_bce", self.cond_sum / self.cond_n.clamp(min=1))
        add("cond_n", self.cond_n)
        add("uncond_bce", self.uncond_sum / valid_n)
        add("event_rate", self.event_n / valid_n)
        brier = self.brier.sum_sq.double() / self.brier.count.double().clamp(min=1)
        add("brier", brier)
        gaps = _bucket_gaps_mmh(_bucket_edges_dbz()).double().view(-1, 1)
        add("crps_contrib", gaps * brier)
        rel = self.reliability
        add("mean_prob", 1 - rel.sum_pred.double().sum(-1) / rel.count.double().sum(-1).clamp(min=1))
        add("ece", (rel.sum_pred - rel.sum_obs).abs().double().sum(-1) / rel.count.double().sum(-1).clamp(min=1))
        # Pooled over all leads: lead_min = "all".
        pooled = {
            "cond_bce": self.cond_sum.sum(1) / self.cond_n.sum(1).clamp(min=1),
            "cond_n": self.cond_n.sum(1),
            "uncond_bce": self.uncond_sum.sum(1) / valid_n.sum(),
            "event_rate": self.event_n.sum(1) / valid_n.sum(),
            "brier": self.brier.sum_sq.double().sum(1) / valid_n.sum(),
            "crps_contrib": (gaps * brier).mean(1),  # CRPS.compute() averages leads
            "mean_prob": 1 - rel.sum_pred.double().sum((1, 2)) / rel.count.double().sum((1, 2)).clamp(min=1),
            "ece": (rel.sum_pred - rel.sum_obs).abs().double().sum((1, 2))
            / rel.count.double().sum((1, 2)).clamp(min=1),
            "loss_contrib": self.loss_contrib(),
        }
        for metric, values in pooled.items():
            for ci, edge in enumerate(EDGES):
                out.append(dict(ckpt=ckpt, split=split, threshold=edge, lead_min="all",
                                metric=metric, value=float(values[ci])))
        return out

    def per_sample_loss(self) -> torch.Tensor:
        return torch.cat(self.sample_loss).nanmean(dim=1)

    def loss_contrib(self) -> torch.Tensor:
        """(C,) -- sums to the loss as logged (mean over samples)."""
        per = torch.cat(self.sample_loss).double()
        n_channels = (~torch.isnan(per)).sum(dim=1, keepdim=True).clamp(min=1)
        return torch.nan_to_num(per / n_channels).mean(dim=0)


def build_datamodule(ckpt_path: str, args) -> RainPro8DataModule:
    hparams = dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)
                   .get("datamodule_hyper_parameters") or {})
    if not hparams:
        sys.exit(f"{ckpt_path} has no datamodule_hyper_parameters")
    accepted = set(inspect.signature(RainPro8DataModule.__init__).parameters) - {"self"}
    dropped = {k: v for k, v in hparams.items() if k not in accepted and v not in (None, "nearest")}
    if dropped:
        sys.exit(f"this branch's RainPro8DataModule does not take {sorted(dropped)} -- the run was "
                 f"trained with them, so evaluate it from the branch it was trained on")
    cfg = {k: v for k, v in hparams.items() if k in accepted}
    if args.data_root:
        cfg["data_root"] = json.loads(args.data_root)
    cfg.update(num_workers=args.num_workers, batch_size=args.batch_size, eval_batch_size=args.batch_size,
               persistent_workers=False, train_jitter_km=0.0)
    if args.num_workers == 0:
        cfg["prefetch_factor"] = None  # DataLoader rejects a prefetch_factor without workers
    for key in ("sample_manifest", "cv_fold", "radar_downsample", "satellite_downsample",
                "train_low_echo_filter", "start_date", "end_date"):
        if key in cfg:
            print(f"  {key}: {cfg[key]}")
    return RainPro8DataModule(**cfg)


def make_loader(dm: RainPro8DataModule, split: str, n_samples: int | None, seed: int, args):
    dataset: RainPro8Dataset = dm._dataloader(split).dataset
    times = list(dm.split_times[split])
    indices = np.arange(len(dataset))
    if n_samples is not None and n_samples < len(dataset):
        indices = np.sort(np.random.default_rng(seed).choice(len(dataset), n_samples, replace=False))
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, pin_memory=args.device.startswith("cuda"))
    return loader, [str(times[i]) for i in indices]


def plot_lines(ax, x, series: dict, ylabel: str, log: bool = False):
    for (ckpt, split), y in series.items():
        ax.plot(x, y, color=CKPT_COLORS.get(ckpt, "#333333"), lw=1.8,
                ls="-" if split == "val" else "--", marker="o", ms=3, label=f"{ckpt} ({split})")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.set_facecolor(SURFACE)


def make_figures(rows: list[dict], samples: dict, ckpts: list[str], out_dir: str, leads: list[int]):
    def get(ckpt, split, metric, lead="all"):
        lookup = {r["threshold"]: r["value"] for r in rows
                  if r["ckpt"] == ckpt and r["split"] == split and r["metric"] == metric and r["lead_min"] == lead}
        return np.array([lookup.get(e, np.nan) for e in EDGES])

    splits = sorted({r["split"] for r in rows}, key=lambda s: s != "val")
    x = np.arange(len(EDGES))
    labels = [f"{e:g}" for e in EDGES]

    def finish(fig, axes, path, title):
        for ax in np.atleast_1d(axes):
            ax.set_xticks(x, labels)
        np.atleast_1d(axes)[-1].set_xlabel("threshold (dBZ)")
        np.atleast_1d(axes)[0].legend(fontsize=8, ncol=2)
        fig.suptitle(title)
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, path), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, height_ratios=(2, 1))
    plot_lines(axes[0], x, {(c, s): get(c, s, "cond_bce") for s in splits for c in ckpts}, "conditional BCE")
    for s, hatch in zip(splits, ("", "//")):
        axes[1].bar(x + (0.2 if s != "val" else -0.2), get(ckpts[0], s, "cond_n"), width=0.4,
                    color="#bbbbbb", hatch=hatch, edgecolor="#666666", label=s)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("conditional pixels")
    axes[1].legend(fontsize=8)
    axes[1].set_facecolor(SURFACE)
    finish(fig, axes, "cond_bce.png", "Conditional BCE per threshold (solid: val, dashed: train subset)")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_lines(ax, x, {(c, "val"): get(c, "val", "loss_contrib") for c in ckpts}, "share of val/loss")
    finish(fig, ax, "loss_contrib.png", "Per-threshold contribution to val/loss")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    plot_lines(axes[0], x, {(c, "val"): get(c, "val", "crps_contrib") for c in ckpts}, "CRPS contribution (mm/h)")
    plot_lines(axes[1], x, {(c, "val"): get(c, "val", "brier") for c in ckpts}, "Brier", log=True)
    plot_lines(axes[2], x, {(c, "val"): get(c, "val", "ece") for c in ckpts}, "ECE", log=True)
    finish(fig, axes, "probabilistic.png", "Unconditional scores per threshold (val)")

    if len(ckpts) >= 2:
        a, b = ckpts[0], ckpts[-1]
        grid = np.stack([get(b, "val", "cond_bce", lead) - get(a, "val", "cond_bce", lead) for lead in leads], axis=1)
        lim = np.nanmax(np.abs(grid)) or 1.0
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_yticks(x, labels)
        step = max(1, len(leads) // 12)
        ax.set_xticks(range(0, len(leads), step), [f"+{leads[i]}" for i in range(0, len(leads), step)])
        ax.set_xlabel("lead time (min)")
        ax.set_ylabel("threshold (dBZ)")
        fig.colorbar(im, ax=ax, label=f"cond BCE: {b} − {a}")
        ax.set_title(f"Conditional BCE difference, {b} − {a} (val)")
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "cond_bce_lead_diff.png"), dpi=150)
        plt.close(fig)

        delta = (samples[(b, "val")] - samples[(a, "val")]).numpy()
        delta = delta[~np.isnan(delta)]
        order = np.sort(delta)[::-1]
        share = np.cumsum(order) / order.sum() if order.sum() != 0 else np.zeros_like(order)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(np.arange(1, len(order) + 1) / len(order) * 100, share * 100, color="#2a78d6", lw=1.8)
        ax.axhline(100, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel("val samples, largest increase first (%)")
        ax.set_ylabel("cumulative share of total change (%)")
        ax.set_title(f"Per-sample val/loss change, {b} − {a}")
        ax.grid(alpha=0.3)
        ax.set_facecolor(SURFACE)
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sample_loss_change.png"), dpi=150)
        plt.close(fig)


def print_summary(rows: list[dict], samples: dict, counts: dict, ckpts: list[str], splits: list[str]):
    def get(ckpt, split, metric):
        return {r["threshold"]: r["value"] for r in rows if r["ckpt"] == ckpt and r["split"] == split
                and r["metric"] == metric and r["lead_min"] == "all"}

    for split in splits:
        print(f"\n== {split}: loss as logged (sum of loss_contrib) / CRPS")
        for c in ckpts:
            loss = sum(get(c, split, "loss_contrib").values())
            crps = sum(get(c, split, "crps_contrib").values())
            print(f"  {c:<10} loss {loss:.4f}   CRPS {crps:.4f}")

    def table(title, metric, split, fmt="{:.4f}"):
        print(f"\n== {title}")
        print("  thr   " + "".join(c.rjust(11) for c in ckpts))
        cols = [get(c, split, metric) for c in ckpts]
        for e in EDGES:
            print(f"  {e:>4g}  " + "".join(fmt.format(col.get(e, float("nan"))).rjust(11) for col in cols))

    n_val = get(ckpts[0], "val", "cond_n")
    print("\n== conditional pixels (val" + (" / train" if "train" in splits else "") + ")")
    n_train = get(ckpts[0], "train", "cond_n") if "train" in splits else {}
    for e in EDGES:
        print(f"  {e:>4g}  {n_val[e]:>14,.0f}" + (f"  {n_train[e]:>14,.0f}" if n_train else ""))
    table("conditional BCE, val", "cond_bce", "val")
    if "train" in splits:
        print("\n== conditional BCE, val − train (generalization gap)")
        print("  thr   " + "".join(c.rjust(11) for c in ckpts))
        for e in EDGES:
            gaps = [get(c, "val", "cond_bce")[e] - get(c, "train", "cond_bce")[e] for c in ckpts]
            print(f"  {e:>4g}  " + "".join(f"{g:+.4f}".rjust(11) for g in gaps))
    table("loss_contrib, val", "loss_contrib", "val")
    table("CRPS contribution (mm/h), val", "crps_contrib", "val")
    table("ECE, val", "ece", "val", fmt="{:.5f}")
    print("\n== mean forecast P(>= thr) / observed rate, val")
    print("  thr   " + "".join(c.rjust(11) for c in ckpts) + "   observed")
    obs = get(ckpts[0], "val", "event_rate")
    cols = [get(c, "val", "mean_prob") for c in ckpts]
    for e in EDGES:
        print(f"  {e:>4g}  " + "".join(f"{col[e]:.5f}".rjust(11) for col in cols) + f"   {obs[e]:.5f}")

    if len(ckpts) >= 2:
        a, b = ckpts[0], ckpts[-1]
        delta = (samples[(b, "val")] - samples[(a, "val")]).numpy()
        ok = ~np.isnan(delta)
        total = delta[ok].sum()
        print(f"\n== per-sample val/loss change, {b} − {a}: mean {delta[ok].mean():+.4f}")
        if total != 0:
            order = np.sort(delta[ok])[::-1]
            for frac in (0.01, 0.05, 0.10, 0.25):
                k = max(1, int(round(frac * len(order))))
                print(f"  top {frac:>4.0%} of samples: {order[:k].sum() / total:6.1%} of the total change")
        cnt = counts["val"].numpy()
        for j, e in enumerate(TOP_EDGES):
            has = (cnt[:, j] > 0) & ok
            if has.any() and total != 0:
                print(f"  samples with any >= {e:g} dBZ pixel: {has.sum() / ok.sum():5.1%} of samples, "
                      f"{delta[has].sum() / total:6.1%} of the change, "
                      f"mean {delta[has].mean():+.4f} vs {delta[ok & ~has].mean() if (ok & ~has).any() else float('nan'):+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="a run's checkpoints/ directory")
    ap.add_argument("--ckpts", default=",".join(CKPT_NAMES),
                    help="comma-separated names in that directory, without .ckpt; missing ones are skipped. "
                         "The first and last are compared in the difference figures")
    ap.add_argument("--data-root", default=None, help="override the checkpoint's data_root (JSON dict)")
    ap.add_argument("--train-samples", type=int, default=1536, help="fixed random train subset size; 0 skips train")
    ap.add_argument("--val-samples", type=int, default=None, help="random val subset (default: all of val)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = [n for n in args.ckpts.split(",") if os.path.exists(os.path.join(args.ckpt_dir, f"{n}.ckpt"))]
    missing = set(args.ckpts.split(",")) - set(ckpts)
    if missing:
        print(f"skipping missing checkpoints: {sorted(missing)}")
    if not ckpts:
        sys.exit(f"no checkpoints found in {args.ckpt_dir}")
    paths = {n: os.path.join(args.ckpt_dir, f"{n}.ckpt") for n in ckpts}
    for n, p in paths.items():
        meta = torch.load(p, map_location="cpu", weights_only=False)
        print(f"  {n}: epoch {meta.get('epoch')}, step {meta.get('global_step')}")
        del meta

    print("datamodule from", paths[ckpts[0]])
    dm = build_datamodule(paths[ckpts[0]], args)
    dm.setup()
    leads = list(dm.sources["target_2km"].offsets_min)

    modules, captures = {}, {}
    for n, p in paths.items():
        module = RainPro8Module.load_from_checkpoint(p, data=dm, map_location=args.device, compile_model=False)
        captures[n] = CaptureLogits(module.model.criterion)
        module.model.criterion = captures[n]
        modules[n] = module.eval().to(args.device)

    splits = ["val"] + (["train"] if args.train_samples > 0 else [])
    rows, samples, counts, sample_rows = [], {}, {}, []
    for split in splits:
        n_samples = args.train_samples if split == "train" else args.val_samples
        loader, times = make_loader(dm, split, n_samples, args.seed, args)
        n_batches = len(loader) if args.limit_batches is None else min(args.limit_batches, len(loader))
        print(f"\n{split}: {len(loader.dataset)} samples, {n_batches} batches", flush=True)
        stats = {n: ThresholdStats(len(leads), captures[n].criterion.weights, args.device) for n in ckpts}
        max_mismatch = 0.0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i == n_batches:
                    break
                batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
                for n, module in modules.items():
                    out = module(batch, EvalRequest(need_loss=True, need_target=True, need_probs=True))
                    mine = stats[n].update(captures[n].logits, out.target, out.probs)
                    max_mismatch = max(max_mismatch, abs(float(mine) - float(out.loss)))
                if (i + 1) % 50 == 0 or i + 1 == n_batches:
                    print(f"  {i + 1}/{n_batches} batches", flush=True)
        # Self-check: the per-sample reduction above must reproduce the loss.
        print(f"  max |recomputed − criterion| batch loss: {max_mismatch:.2e}")
        if max_mismatch > 1e-3:
            print("  !! recomputed loss disagrees with OrdinalConsistentLoss -- per-sample numbers are suspect")
        for n in ckpts:
            rows += stats[n].rows(n, split, leads)
            samples[(n, split)] = stats[n].per_sample_loss()
        counts[split] = torch.cat(stats[ckpts[0]].sample_counts)
        for n in ckpts:
            for j, loss in enumerate(samples[(n, split)].tolist()):
                sample_rows.append(dict(ckpt=n, split=split, sample=j, init_time=times[j], loss=loss,
                                        **{f"n_ge{e:g}": int(counts[split][j, k]) for k, e in enumerate(TOP_EDGES)}))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in (("per_threshold.csv", rows), ("per_sample.csv", sample_rows)):
        with open(os.path.join(args.out_dir, name), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print_summary(rows, samples, counts, ckpts, splits)
    make_figures(rows, samples, ckpts, args.out_dir, leads)
    print(f"\nwrote {args.out_dir}/: per_threshold.csv, per_sample.csv, figures")


if __name__ == "__main__":
    main()
