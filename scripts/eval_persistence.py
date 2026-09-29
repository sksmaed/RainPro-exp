"""Score a checkpoint against 2 km persistence on a whole split, per lead time
and threshold.

Persistence = the t0 QPESUMS frame on the target's 2 km grid, repeated for
every lead time (read through a separate single-source Dataset, exactly as in
scripts/infer_visualize.py, so the model-facing datamodule keeps its 36
target steps). Both forecasts go through the same metric classes the
LightningModule logs, with the same NaN-target masking:

  * CSI, FSS (every window), POD / FAR / FBI -- deterministic. The model side
    uses `EvalOutputs.forecast` (the 0.5-threshold pick, values are bucket
    minima, so "forecast >= 35" means the 37 dBZ bucket or above).
  * CRPS, Brier -- probabilistic. Persistence enters as a step CDF (all mass
    on its own t0 value), so these compare the model's full distribution with
    a deterministic baseline on the same footing. Where the 0.5 forecast
    scores CSI = 0, these still say whether the model has probabilistic skill.

Writes every number to a long-format CSV (source, metric, threshold, window,
lead_min, value) and prints a summary at representative lead times.

The split comes from `RainPro8DataModule.setup()` with the given config, so it
is the current (STA_H8-filtered) val/test set -- scores are comparable across
checkpoints evaluated with this script, not with numbers from older runs'
W&B logs. Evaluate a checkpoint with the input pipeline it was trained with.

Usage (GPU node recommended; add --limit-batches 20 for a quick check):
    python scripts/eval_persistence.py \\
        --config rainpro8.yml \\
        --ckpt runs/rainpro8_2021_obs_only/checkpoints/best.ckpt \\
        --data-root '{"qpesums": "...", "sta_h8": "...Q1.zarr,...Q2.zarr,..."}' \\
        --variable-aliases '{"max_dbz": "MaxDBZ"}' \\
        --out-csv figures/eval_val_best.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import inspect
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainpro.data.rainpro8_datamodule import RainPro8DataModule  # noqa: E402
from rainpro.data.rainpro8_dataset import RainPro8Dataset  # noqa: E402
from rainpro.loss.ordinal_consistent import taiwan_dbz_buckets  # noqa: E402
from rainpro.metrics.contingency import ContingencyMetrics  # noqa: E402
from rainpro.metrics.csi import CriticalSuccessIndex  # noqa: E402
from rainpro.metrics.fss import FractionsSkillScore  # noqa: E402
from rainpro.metrics.probabilistic import CRPS, BrierScore  # noqa: E402
from rainpro.modules.rainpro8 import CSI_THRESHOLDS_DBZ, RainPro8Module  # noqa: E402
from rainpro.modules.utils import EvalOutputs, EvalRequest  # noqa: E402

SUMMARY_LEADS = (10, 20, 30, 60, 120, 180, 360)


class WithPersistence(Dataset):
    """Adds the t0 target frame, (1, 1, H, W), as `persistence` to each sample."""

    def __init__(self, base: RainPro8Dataset, persistence: RainPro8Dataset):
        assert len(base) == len(persistence)
        self.base, self.persistence = base, persistence

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index]
        sample["persistence"] = self.persistence[index]["target_2km"]
        return sample


def make_metrics(num_lead_times: int) -> dict:
    return {
        "csi": CriticalSuccessIndex(num_lead_times, CSI_THRESHOLDS_DBZ),
        "contingency": ContingencyMetrics(num_lead_times, CSI_THRESHOLDS_DBZ),
        "fss": FractionsSkillScore(num_lead_times, CSI_THRESHOLDS_DBZ),
        "crps": CRPS(num_lead_times),
        "brier": BrierScore(num_lead_times),
    }


def load_datamodule(args) -> RainPro8DataModule:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["data"]
    cfg = cfg.get("init_args", cfg)  # tolerate a class_path/init_args layout
    accepted = set(inspect.signature(RainPro8DataModule.__init__).parameters) - {"self"}
    cfg = {k: v for k, v in cfg.items() if k in accepted}
    cfg["data_root"] = json.loads(args.data_root)
    if args.variable_aliases:
        cfg["variable_aliases"] = json.loads(args.variable_aliases)
    cfg["num_workers"] = args.num_workers
    cfg["batch_size"] = cfg["eval_batch_size"] = args.batch_size
    return RainPro8DataModule(**cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="rainpro8.yml or a run's saved config.yml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True, help="same JSON dict as --data.data_root")
    ap.add_argument("--variable-aliases", default=None)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--out-csv", default="eval_persistence.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dm = load_datamodule(args)
    dm.setup()
    target_spec = dm.sources["target_2km"]
    leads = list(target_spec.offsets_min)
    init_times = dm.split_times[args.split]
    print(f"{args.split}: {len(init_times)} init_times", flush=True)

    base = dm._dataloader(args.split).dataset
    persistence = RainPro8Dataset(
        data_root=dm.data_root,
        sources={"target_2km": dataclasses.replace(target_spec, offsets_min=(0,))},
        init_times=init_times,
        center_lat=dm.center_lat,
        center_lon=dm.center_lon,
        jitter_km=0.0,
        variable_aliases=dm.variable_aliases,
        latlon_names=dm.latlon_names,
    )
    loader = DataLoader(
        WithPersistence(base, persistence),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.device.startswith("cuda"),
    )

    module = RainPro8Module.load_from_checkpoint(
        args.ckpt, data=dm, map_location=args.device, compile_model=False
    ).eval().to(args.device)

    metrics = {src: make_metrics(len(leads)) for src in ("model", "persistence")}
    for group in metrics.values():
        for m in group.values():
            m.to(args.device)
    edges = torch.tensor([b.min for b in taiwan_dbz_buckets()], device=args.device).view(1, 1, -1, 1, 1)

    n_batches = len(loader) if args.limit_batches is None else min(args.limit_batches, len(loader))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i == n_batches:
                break
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
            target = batch["target_2km"]  # (B, T, 1, H, W)

            out = module(batch, EvalRequest(need_forecast=True, need_target=True, need_probs=True))
            model_out = EvalOutputs(forecast=out.forecast, target=target, probs=out.probs)

            # Out-of-coverage t0 pixels carry no forecast: call them "no echo"
            # (target-NaN pixels are masked by every metric anyway).
            t0 = torch.nan_to_num(batch["persistence"], nan=0.0)  # (B, 1, 1, H, W)
            pers = t0.expand_as(target)
            # Step CDF: F(edge) = 1 if the persisted value is <= edge -- the
            # same "<= edge" convention CRPS/Brier use for the target.
            pers_out = EvalOutputs(forecast=pers, target=target, probs=(pers <= edges).float())

            for src, eo in (("model", model_out), ("persistence", pers_out)):
                for m in metrics[src].values():
                    m.update(eo)
            if (i + 1) % 20 == 0 or i + 1 == n_batches:
                print(f"  {i + 1}/{n_batches} batches", flush=True)

    rows: list[dict] = []

    def add(src, metric, table, thresholds=None, window=""):
        table = table.detach().cpu()
        if table.ndim == 1:
            for li, lead in enumerate(leads):
                rows.append(dict(source=src, metric=metric, threshold="", window=window,
                                 lead_min=lead, value=float(table[li])))
            return
        for ti, th in enumerate(thresholds):
            for li, lead in enumerate(leads):
                rows.append(dict(source=src, metric=metric, threshold=th, window=window,
                                 lead_min=lead, value=float(table[ti, li])))

    brier_edges = [b.min for b in taiwan_dbz_buckets()]
    for src, group in metrics.items():
        add(src, "CSI", group["csi"]._compute(reduce_mean=False), CSI_THRESHOLDS_DBZ)
        for name, table in group["contingency"].full().items():
            add(src, name, table, CSI_THRESHOLDS_DBZ)
        for name, table in group["fss"].full().items():
            add(src, "FSS", table, CSI_THRESHOLDS_DBZ, window=name.removeprefix("FSS_w"))
        add(src, "CRPS", group["crps"].full()["CRPS"])
        add(src, "Brier", group["brier"].full()["Brier"], brier_edges)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "metric", "threshold", "window", "lead_min", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out_csv} ({len(rows)} rows, {n_batches}/{len(loader)} batches)")

    lookup = {(r["source"], r["metric"], r["threshold"], r["window"], r["lead_min"]): r["value"] for r in rows}
    fss_window = str(group["fss"].windows[-1])
    columns = [  # (label, metric, threshold, window)
        ("CSI20", "CSI", 20.0, ""), ("CSI30", "CSI", 30.0, ""), ("CSI35", "CSI", 35.0, ""),
        (f"FSS20w{fss_window}", "FSS", 20.0, fss_window), ("FBI20", "FBI", 20.0, ""),
        ("FBI35", "FBI", 35.0, ""), ("POD35", "POD", 35.0, ""), ("CRPS", "CRPS", "", ""),
    ]
    print("\nmodel / persistence")
    print("  lead  " + "".join(label.rjust(15) for label, *_ in columns))
    for lead in SUMMARY_LEADS:
        if lead not in leads:
            continue
        cells = []
        for _, metric, th, window in columns:
            m = lookup.get(("model", metric, th, window, lead), float("nan"))
            p = lookup.get(("persistence", metric, th, window, lead), float("nan"))
            cells.append(f"{m:.3f}/{p:.3f}".rjust(15))
        print(f"  +{lead:<4}" + "".join(cells))


if __name__ == "__main__":
    main()
