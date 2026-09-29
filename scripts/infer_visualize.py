"""Run trained checkpoints on one init_time and write one diagnostic figure per
checkpoint.

Each figure is a lead-time x view grid with six fixed columns:

    expected dBZ | 0.5 forecast | P(>20 dBZ) | P(>34 dBZ) | persistence | ground truth

and one row per `--lead-times` entry (default +10/+20/+30 min, +1/+2/+3/+6 h:
the first three probe short-range detail, where the 4 km radar input vs. 2 km
target resolution gap would show; the rest show how uncertainty spreads).

What each view answers
----------------------
`RainPro` is ordinal-classification, not regression: `RainPro.predict` returns
`probs = 1 - cumprod(sigmoid(logits))`, i.e. the CDF at each bucket boundary,
so the exceedance curve is `S_k = 1 - probs = P(Y > t_k)` over the 16
`taiwan_dbz_buckets` boundaries [5, 10, 15, 20, 25, 28, ..., 55, 60] dBZ.

  * expected dBZ -- the standard bin-probability x representative-value sum
    (`expected_dbz` below). Averages over uncertainty, so a 50/50 "0 or 30
    dBZ" pixel reads ~15 dBZ: edges look soft and peaks fade with lead time
    even when the model is only unsure *where*, not *how strong*.
  * 0.5 forecast -- `EvalOutputs.forecast`, `Threshold`'s pick of the largest
    boundary with P(Y > t_k) > 0.5: a median-like, discretized forecast.
    Crisper edges than the expectation, but peaks are usually *lower*
    (P(>40 dBZ) rarely beats 0.5), so crisper != closer to GT intensity.
  * P(>t) maps -- separate "weak but correctly placed" from "strong but
    position-uncertain": a wide field of 0.1-0.3 at 34 dBZ means the model
    expects strong echo somewhere around there; ~0 everywhere means it
    really predicts it weaker. Thresholds must be bucket boundaries (35 is
    not one -- 34 is the nearest), see `--prob-thresholds`.
  * persistence -- the t0 QPESUMS frame at 2 km, repeated for every lead
    time. Read through a separate single-source Dataset (target_2km with
    offset 0) so the model-facing datamodule keeps its 36 target steps --
    `frames_out` sizes the network, so changing that spec would build a
    checkpoint-incompatible model.

`expected_dbz` representative values are judgement calls, both conservative:
  * LOW_BIN_DBZ = 0, not the bin's 2.5 dBZ midpoint -- QPESUMS clear sky reads
    ~0, and a 2.5 dBZ floor would tint the entire clear domain in a way the
    ground-truth panel doesn't have.
  * TAIL_DBZ = 60, the top boundary rather than an extrapolation past it, so
    the open-ended top bin can't inflate peaks beyond what the model can
    actually resolve.

Also prints two per-lead-time tables for this single case -- anecdotal, not a
substitute for validation-set scores (scripts/eval_persistence.py):
  * CSI, 0.5 forecast vs. persistence, against GT.
  * Probability mass vs. event area, `--mass-thresholds`: sum over pixels of
    P(Y > t) x 4 km^2 against the GT area >= t (both over GT-valid pixels).
    Mass close to the GT area but a diffuse P map -> the model knows how much
    strong echo there is, just not where. Mass far below it -> it genuinely
    predicts too little strong echo.

Usage:
    python scripts/infer_visualize.py \\
        --init-time 2023-06-01T06:00 \\
        --data-root '{"qpesums": "...", "sta_h8": "/work/kilin1203/datasets/STA_H8/2023/06/01"}' \\
        --variable-aliases '{"max_dbz": "MaxDBZ"}' \\
        --out figures/inference_20230601_0600.png
    # -> figures/inference_20230601_0600_best_crps.png, ..._best_loss.png, ..._last.png

Note `sta_h8` may point at a raw `.btp` directory (no zarr conversion needed --
`RainPro8Dataset._get_store` falls back to the on-demand reader). Point it at
the narrowest directory that covers the init_time; the fallback walks whatever
tree it's given, so a whole-year root costs a long scan for nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainpro.data.rainpro8_dataset import RainPro8Dataset  # noqa: E402
from rainpro.data.rainpro8_datamodule import RainPro8DataModule  # noqa: E402
from rainpro.loss.ordinal_consistent import taiwan_dbz_buckets  # noqa: E402
from rainpro.modules.rainpro8 import RainPro8Module  # noqa: E402
from rainpro.modules.utils import EvalRequest  # noqa: E402

LOW_BIN_DBZ = 0.0  # representative value for "at or below the lowest boundary"
TAIL_DBZ = 60.0  # representative value for the open-ended top bin
PLOT_BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
PROB_BOUNDS = np.linspace(0, 1, 11)
BUCKET_EDGES = [b.min for b in taiwan_dbz_buckets()]


def expected_dbz(probs: torch.Tensor) -> torch.Tensor:
    """(B, T, K, H, W) CDF -> (B, T, H, W) expected dBZ. See module docstring."""
    exceedance = 1.0 - probs  # S_k = P(Y > t_k), decreasing in k
    boundaries = torch.tensor(BUCKET_EDGES, dtype=exceedance.dtype, device=exceedance.device)

    p_low = 1.0 - exceedance[:, :, 0]
    p_mid = exceedance[:, :, :-1] - exceedance[:, :, 1:]
    p_tail = exceedance[:, :, -1]

    v_mid = ((boundaries[:-1] + boundaries[1:]) / 2).view(1, 1, -1, 1, 1)
    return p_low * LOW_BIN_DBZ + (p_mid * v_mid).sum(dim=2) + p_tail * TAIL_DBZ


def csi(forecast: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    """Single-frame CSI, pixels where either field is NaN excluded."""
    valid = np.isfinite(forecast) & np.isfinite(truth)
    f = (forecast >= threshold) & valid
    o = (truth >= threshold) & valid
    denom = (f | o).sum()
    return float((f & o).sum() / denom) if denom else float("nan")


def load_module(ckpt_path: str, datamodule: RainPro8DataModule, device: str) -> RainPro8Module:
    # `data` is excluded from the checkpoint's hparams (`save_hyperparameters(
    # ignore="data")`), so it has to be supplied here; everything else
    # (max_epochs, dims, dropout, ...) comes back from the checkpoint.
    # compile_model=False: one forward pass never repays torch.compile's
    # compile time (and CPU compile needs a working C++ toolchain).
    module = RainPro8Module.load_from_checkpoint(
        ckpt_path, data=datamodule, map_location=device, compile_model=False
    )
    return module.eval().to(device)


def default_ckpts(ckpt_dir: str) -> list[str]:
    """best_crps (model selection) / best_loss / last; runs from before
    `checkpoint_monitors` existed only have best.ckpt + last.ckpt."""
    names = ["best_crps.ckpt", "best_loss.ckpt", "last.ckpt"]
    if not os.path.isfile(os.path.join(ckpt_dir, "best_crps.ckpt")):
        names = ["best.ckpt", "last.ckpt"]
    return [os.path.join(ckpt_dir, n) for n in names]


def plot_one(
    out_path: str,
    ckpt_label: str,
    init_time: np.datetime64,
    lead_times: list[int],
    columns: list[tuple[str, np.ndarray, str]],
) -> None:
    """`columns`: (title, (n_lead, H, W) field, "dbz" | "prob")."""
    dbz_cmap = mpl.colormaps["turbo"].resampled(len(PLOT_BOUNDS) - 1)
    dbz_cmap.set_bad("0.85")  # NaN (no radar coverage) reads as grey, not as 0 dBZ
    dbz_norm = mpl.colors.BoundaryNorm(PLOT_BOUNDS, dbz_cmap.N)
    prob_cmap = mpl.colormaps["viridis"].resampled(len(PROB_BOUNDS) - 1)
    prob_cmap.set_bad("0.85")
    prob_norm = mpl.colors.BoundaryNorm(PROB_BOUNDS, prob_cmap.N)

    n_rows = len(lead_times)
    fig, axes = plt.subplots(
        n_rows, len(columns), figsize=(3.1 * len(columns), 3.0 * n_rows + 1.0),
        squeeze=False, layout="constrained",
    )
    for row, lead in enumerate(lead_times):
        for col, (title, field, kind) in enumerate(columns):
            ax = axes[row][col]
            cmap, norm = (dbz_cmap, dbz_norm) if kind == "dbz" else (prob_cmap, prob_norm)
            ax.imshow(np.ma.masked_invalid(field[row]), cmap=cmap, norm=norm, origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                label = f"+{lead} min" if lead < 60 else f"+{lead // 60} h" if lead % 60 == 0 else f"+{lead} min"
                ax.set_ylabel(label, fontsize=11)

    dbz_axes = [axes[r][c] for r in range(n_rows) for c, (_, _, k) in enumerate(columns) if k == "dbz"]
    prob_axes = [axes[r][c] for r in range(n_rows) for c, (_, _, k) in enumerate(columns) if k == "prob"]
    fig.colorbar(mpl.cm.ScalarMappable(norm=dbz_norm, cmap=dbz_cmap), ax=dbz_axes,
                 orientation="horizontal", fraction=0.02, aspect=45, pad=0.01,
                 label="reflectivity (dBZ)", ticks=PLOT_BOUNDS)
    fig.colorbar(mpl.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap), ax=prob_axes,
                 orientation="horizontal", fraction=0.02, aspect=30, pad=0.01,
                 label="exceedance probability", ticks=PROB_BOUNDS[::2])
    fig.suptitle(f"RainPro-8 TW  |  {ckpt_label}  |  init {np.datetime_as_string(init_time, unit='m')}",
                 fontsize=13)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # constrained_layout already handles spacing; bbox_inches="tight" would fight it
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-time", required=True, help="e.g. 2023-06-01T06:00")
    ap.add_argument("--data-root", required=True, help="same JSON dict as --data.data_root")
    ap.add_argument("--variable-aliases", default="{}")
    ap.add_argument("--ckpt", action="append", default=None,
                     help="repeatable; defaults to best_crps + best_loss + last (or best + last "
                          "for older runs) under --ckpt-dir")
    ap.add_argument("--ckpt-dir", default="runs/rainpro8_2021_obs_only/checkpoints")
    ap.add_argument("--lead-times", default="10,20,30,60,120,180,360",
                     help="comma-separated minutes, multiples of 10 in 10..360 (one figure row each)")
    ap.add_argument("--prob-thresholds", default="20,34",
                     help="comma-separated dBZ for the P(>t) columns; must be bucket boundaries "
                          f"{BUCKET_EDGES}")
    ap.add_argument("--mass-thresholds", default="20,34,40",
                     help="comma-separated dBZ (bucket boundaries) for the probability-mass table")
    ap.add_argument("--out", default="inference.png",
                     help="output path stem; one file per checkpoint, <stem>_<ckpt name>.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # These two decide `in_dims`, so they MUST match how the checkpoint was
    # trained -- otherwise the first conv of each tier has the wrong channel
    # count and load_from_checkpoint dies on a shape mismatch deep in the
    # state_dict. Defaults are the obs-only arm this run trained.
    ap.add_argument("--include-satellite", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-gfs", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    ckpts = args.ckpt or default_ckpts(args.ckpt_dir)
    missing = [c for c in ckpts if not os.path.isfile(c)]
    if missing:
        raise SystemExit(f"checkpoint(s) not found: {missing}")

    data_root = json.loads(args.data_root)
    variable_aliases = json.loads(args.variable_aliases)
    init_time = np.datetime64(args.init_time)

    # Built only for its `sources` / `frames_out`, which RainPro8Module needs to
    # size the network. `setup()` is deliberately NOT called -- that scans the
    # archives to filter split times, which inference doesn't need.
    datamodule = RainPro8DataModule(
        data_root=data_root,
        start_date="2021-01-01",
        end_date="2022-01-01",
        include_satellite=args.include_satellite,
        include_gfs=args.include_gfs,
        variable_aliases=variable_aliases,
    )
    target_spec = datamodule.sources["target_2km"]

    lead_times = [int(x) for x in args.lead_times.split(",")]
    bad = [lt for lt in lead_times if lt not in target_spec.offsets_min]
    if bad:
        raise SystemExit(f"--lead-times {bad} not in the target's offsets {list(target_spec.offsets_min)}")
    lead_idx = [list(target_spec.offsets_min).index(lt) for lt in lead_times]

    prob_thresholds = [float(x) for x in args.prob_thresholds.split(",")]
    bad = [t for t in prob_thresholds if t not in BUCKET_EDGES]
    if bad:
        raise SystemExit(f"--prob-thresholds {bad} are not bucket boundaries {BUCKET_EDGES}")
    # probs[:, :, k] is the CDF at BUCKET_EDGES[k], so 1 - probs is P(Y > edge)
    prob_idx = [BUCKET_EDGES.index(t) for t in prob_thresholds]
    mass_thresholds = [float(x) for x in args.mass_thresholds.split(",")]
    bad = [t for t in mass_thresholds if t not in BUCKET_EDGES]
    if bad:
        raise SystemExit(f"--mass-thresholds {bad} are not bucket boundaries {BUCKET_EDGES}")

    dataset = RainPro8Dataset(
        data_root=data_root,
        sources=datamodule.sources,
        init_times=[init_time],
        jitter_km=0.0,  # inference: no augmentation, canvas centred as configured
        variable_aliases=variable_aliases,
    )
    # Persistence: a separate single-source Dataset reading the target at t0.
    # Same spec name/store as target_2km, so identical 2 km grid, no-echo and
    # missing handling, and raw (unnormalized) dBZ.
    persistence_dataset = RainPro8Dataset(
        data_root=data_root,
        sources={"target_2km": dataclasses.replace(target_spec, offsets_min=(0,))},
        init_times=[init_time],
        jitter_km=0.0,
        variable_aliases=variable_aliases,
    )

    print(f"building input sample for {init_time} ...", flush=True)
    batch = {k: v.unsqueeze(0).to(args.device) for k, v in dataset[0].items()}
    truth = batch["target_2km"][0, lead_idx, 0].cpu().numpy()  # (n_lead, H, W), NaN where missing
    if np.isnan(truth).all():
        print("!! ground truth is entirely NaN -- QPESUMS has no coverage for this init_time")
    t0 = persistence_dataset[0]["target_2km"][0, 0].numpy()
    if np.isnan(t0).all():
        print("!! t0 QPESUMS frame is entirely NaN -- persistence column will be blank")
    persistence = np.broadcast_to(t0, truth.shape)

    stem, ext = os.path.splitext(args.out)
    for ckpt in ckpts:
        ckpt_label = os.path.splitext(os.path.basename(ckpt))[0]
        print(f"\nrunning {ckpt} on {args.device} ...", flush=True)
        module = load_module(ckpt, datamodule, args.device)
        with torch.no_grad():
            out = module(batch, EvalRequest(need_forecast=True, need_probs=True))
        del module
        assert out.probs is not None

        expected = expected_dbz(out.probs)[0, lead_idx].cpu().numpy()
        forecast = out.forecast[0, lead_idx, 0].cpu().numpy()
        exceed = (1.0 - out.probs[0, lead_idx]).cpu().numpy()  # (n_lead, K, H, W)

        columns: list[tuple[str, np.ndarray, str]] = [
            ("expected dBZ", expected, "dbz"),
            ("0.5 forecast", forecast, "dbz"),
            *[(f"P(>{t:g} dBZ)", exceed[:, k], "prob") for t, k in zip(prob_thresholds, prob_idx)],
            ("persistence (t0)", persistence, "dbz"),
            ("ground truth", truth, "dbz"),
        ]
        out_path = f"{stem}_{ckpt_label}{ext or '.png'}"
        plot_one(out_path, ckpt_label, init_time, lead_times, columns)
        print(f"wrote {out_path}")

        header = "  lead    " + "  ".join(f"CSI{t:g} fcst/pers" for t in prob_thresholds)
        print(header)
        for row, lt in enumerate(lead_times):
            cells = [
                f"{csi(forecast[row], truth[row], t):.3f}/{csi(persistence[row], truth[row], t):.3f}"
                .rjust(len(f"CSI{t:g} fcst/pers"))
                for t in prob_thresholds
            ]
            print(f"  +{lt:<4}   " + "  ".join(cells))

        # Probability mass vs. event area, over GT-valid pixels only. The
        # target is 2 km, so each pixel is 4 km^2. Compared against GT >= t
        # because that is the event channel t was trained on: the loss's
        # `Bucketize(right=True)` encodes a target of exactly t as exceeding t.
        px_km2 = target_spec.resolution_km**2
        valid = np.isfinite(truth)  # (n_lead, H, W)
        print("  lead    " + "  ".join(f"sumP>{t:g} / GT>={t:g} km2".rjust(24) for t in mass_thresholds))
        for row, lt in enumerate(lead_times):
            cells = []
            for t in mass_thresholds:
                k = BUCKET_EDGES.index(t)
                mass = float(exceed[row, k][valid[row]].sum() * px_km2)
                area = float((truth[row][valid[row]] >= t).sum() * px_km2)
                ratio = f"({mass / area:.2f}x)" if area else "(  -  )"
                cells.append(f"{mass:7.0f} / {area:7.0f} {ratio:>8}".rjust(24))
            print(f"  +{lt:<4}   " + "  ".join(cells))


if __name__ == "__main__":
    main()
