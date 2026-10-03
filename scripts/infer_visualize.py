"""Run trained checkpoints on one init_time and write three kinds of figure.
All map figures put lead time along the columns (default +10/+20/+30 min,
+1/+2/+3/+6 h, see `--lead-times`) and what is being compared down the rows.

  <stem>_overview.png   ground truth + optical flow + one row per checkpoint
                        (expected dBZ).
  <stem>_<ckpt>.png     one per checkpoint, rows: ground truth | persistence |
                        optical flow | expected dBZ | 0.5 forecast |
                        P(>20 dBZ) | P(>34 dBZ).
  <stem>_hist.png       the predicted distribution at two grid points, picked
                        automatically at `--hist-lead`: where GT is largest,
                        and where it is smallest (`--min-point`). Top row: GT
                        and each checkpoint's expected dBZ with both points
                        marked; below, per point, a zoom on the GT around it
                        and each checkpoint's per-bucket probabilities, with
                        the GT bucket outlined and GT / predicted values noted.

Colour: 0-5 dBZ and probability < 0.02 are white (no echo / no signal); grey
is outside radar coverage (NaN).

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
  * optical flow -- `rainpro.baselines.optical_flow`, the extrapolation
    baseline `RainPro8Module.test_step` scores under `test_optflow/`: motion
    from the sample's own `radar_4km` frames, t0 advected along it. The
    honest bar for "does the model beat moving the echo along": persistence
    only shows what standing still costs.

`expected_dbz` representative values are judgement calls, both conservative:
  * LOW_BIN_DBZ = 0, not the bin's 2.5 dBZ midpoint -- QPESUMS clear sky reads
    ~0, and a 2.5 dBZ floor would tint the entire clear domain in a way the
    ground-truth panel doesn't have.
  * TAIL_DBZ = 60, the top boundary rather than an extrapolation past it, so
    the open-ended top bin can't inflate peaks beyond what the model can
    actually resolve.

Also prints two per-lead-time tables for this single case -- anecdotal, not a
substitute for validation-set scores (scripts/eval_persistence.py):
  * CSI, 0.5 forecast vs. persistence vs. optical flow, against GT.
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
    # -> figures/inference_20230601_0600_overview.png, ..._hist.png,
    #    ..._best_crps.png, ..._best_loss.png, ..._last.png

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
from rainpro.baselines.optical_flow import OpticalFlowBaseline  # noqa: E402
from rainpro.data.rainpro8_dataset import RainPro8Dataset  # noqa: E402
from rainpro.data.rainpro8_datamodule import RainPro8DataModule  # noqa: E402
from rainpro.loss.ordinal_consistent import taiwan_dbz_buckets  # noqa: E402
from rainpro.modules.rainpro8 import RainPro8Module  # noqa: E402
from rainpro.modules.utils import EvalRequest  # noqa: E402

LOW_BIN_DBZ = 0.0  # representative value for "at or below the lowest boundary"
TAIL_DBZ = 60.0  # representative value for the open-ended top bin
PLOT_BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# Fine at the low end on purpose: forecast probability for strong echo is
# mostly spread thin (0.02-0.2), which a linear 0.1-step scale renders as blank.
PROB_BOUNDS = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BUCKET_EDGES = [b.min for b in taiwan_dbz_buckets()]
NAN_GREY = "0.72"  # outside radar coverage; must stay distinct from white (0 dBZ)


def dbz_colormap() -> tuple[mpl.colors.Colormap, mpl.colors.Normalize]:
    """turbo with the first bin (0-5 dBZ, i.e. no echo) white."""
    base = mpl.colormaps["turbo"].resampled(len(PLOT_BOUNDS) - 1)
    cmap = mpl.colors.ListedColormap(["white"] + [base(i) for i in range(1, base.N)])
    cmap.set_bad(NAN_GREY)
    cmap.set_under("white")
    return cmap, mpl.colors.BoundaryNorm(PLOT_BOUNDS, cmap.N)


def prob_colormap() -> tuple[mpl.colors.Colormap, mpl.colors.Normalize]:
    """Light-to-dark sequential, first bin (< 0.02) white."""
    n = len(PROB_BOUNDS) - 1
    base = mpl.colormaps["YlGnBu"]
    cmap = mpl.colors.ListedColormap(["white"] + [base(x) for x in np.linspace(0.12, 1.0, n - 1)])
    cmap.set_bad(NAN_GREY)
    return cmap, mpl.colors.BoundaryNorm(PROB_BOUNDS, cmap.N)


def lead_label(lead: int) -> str:
    return f"+{lead // 60} h" if lead >= 60 and lead % 60 == 0 else f"+{lead} min"


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


def plot_grid(
    out_path: str,
    title: str,
    lead_times: list[int],
    rows: list[tuple[str, np.ndarray, str]],
) -> None:
    """Map grid, lead time along the columns. `rows`: (label, (n_lead, H, W)
    field, "dbz" | "prob")."""
    dbz_cmap, dbz_norm = dbz_colormap()
    prob_cmap, prob_norm = prob_colormap()
    n_cols = len(lead_times)
    fig, axes = plt.subplots(
        len(rows), n_cols, figsize=(2.5 * n_cols + 1.4, 2.5 * len(rows) + 0.6),
        squeeze=False, layout="constrained",
    )
    for r, (label, field, kind) in enumerate(rows):
        cmap, norm = (dbz_cmap, dbz_norm) if kind == "dbz" else (prob_cmap, prob_norm)
        for c, lead in enumerate(lead_times):
            ax = axes[r][c]
            ax.imshow(np.ma.masked_invalid(field[c]), cmap=cmap, norm=norm, origin="lower",
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(lead_label(lead), fontsize=12)
            if c == 0:
                ax.set_ylabel(label, fontsize=11)

    dbz_axes = [axes[r][c] for r, (_, _, k) in enumerate(rows) if k == "dbz" for c in range(n_cols)]
    prob_axes = [axes[r][c] for r, (_, _, k) in enumerate(rows) if k == "prob" for c in range(n_cols)]
    fig.colorbar(mpl.cm.ScalarMappable(norm=dbz_norm, cmap=dbz_cmap), ax=dbz_axes,
                 orientation="vertical", fraction=0.015, aspect=35, pad=0.01,
                 label="reflectivity (dBZ)", ticks=PLOT_BOUNDS)
    if prob_axes:
        fig.colorbar(mpl.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap), ax=prob_axes,
                     orientation="vertical", fraction=0.015, aspect=20, pad=0.01,
                     label="exceedance probability", ticks=PROB_BOUNDS)
    fig.suptitle(title, fontsize=13)
    save(fig, out_path)


def save(fig, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # constrained_layout already handles spacing; bbox_inches="tight" would fight it
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def pick_points(truth: np.ndarray, min_point: str) -> dict[str, tuple[int, int]]:
    """GT argmax, and the GT minimum -- over echo pixels (GT > 0) for
    `min_point="echo"`, over every covered pixel for "any". Many pixels tie at
    the minimum (clear sky is all 0 dBZ), so the tie is broken by distance to
    the max point: the most relevant low-value pixel is the one nearest the
    storm, not an arbitrary corner of the canvas."""
    valid = np.isfinite(truth)
    if not valid.any():
        raise SystemExit("ground truth at --hist-lead is entirely NaN; pick another lead or init time")
    masked = np.where(valid, truth, -np.inf)
    max_rc = np.unravel_index(np.argmax(masked), truth.shape)

    candidates = valid & (truth > 0) if min_point == "echo" else valid
    if not candidates.any():
        print("!! no echo pixel at --hist-lead; min point falls back to any covered pixel")
        candidates = valid
    lowest = truth[candidates].min()
    rr, cc = np.nonzero(candidates & (truth == lowest))
    nearest = np.argmin((rr - max_rc[0]) ** 2 + (cc - max_rc[1]) ** 2)
    return {"GT max": (int(max_rc[0]), int(max_rc[1])), "GT min": (int(rr[nearest]), int(cc[nearest]))}


def bin_probabilities(exceed_k: np.ndarray) -> np.ndarray:
    """(K,) exceedance P(Y >= t_k) at one pixel -> (K + 1,) probability per
    bucket: [< t_0, [t_0, t_1), ..., >= t_last]. Same decomposition as
    `expected_dbz`."""
    return np.concatenate([[1.0 - exceed_k[0]], exceed_k[:-1] - exceed_k[1:], [exceed_k[-1]]])


def bucket_of(value: float) -> int:
    """Index into `bin_probabilities`' output for a dBZ value. `>=` edges,
    matching the loss's `Bucketize(right=True)`."""
    return int(np.sum(np.asarray(BUCKET_EDGES) <= value))


def plot_hist(
    out_path: str,
    init_time: np.datetime64,
    lead: int,
    truth: np.ndarray,
    points: dict[str, tuple[int, int]],
    ckpt_results: list[tuple[str, dict]],
    zoom_px: int = 24,
) -> None:
    """Top row: where the points are (GT + each checkpoint's expected dBZ).
    One row per point below: GT zoom around it + per-checkpoint histogram."""
    dbz_cmap, dbz_norm = dbz_colormap()
    labels = ["<5"] + [f"{a:g}-{b:g}" for a, b in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])] + [f">={BUCKET_EDGES[-1]:g}"]
    bar_mid = [2.5] + [(a + b) / 2 for a, b in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])] + [BUCKET_EDGES[-1]]
    bar_colors = [dbz_cmap(dbz_norm(v)) for v in bar_mid]
    # (marker, colour, label offset in points): labels go above / below so they
    # stay readable when the two points are close together.
    markers = {"GT max": ("o", "black", (6, 6)), "GT min": ("s", "magenta", (6, -14))}

    n_cols = 1 + len(ckpt_results)
    fig = plt.figure(figsize=(4.2 * n_cols, 4.0 * (1 + len(points))), layout="constrained")
    grid = fig.add_gridspec(1 + len(points), n_cols)

    def mark(ax, only=None, dx=0, dy=0):
        for name, (r, c) in points.items():
            if only is not None and name != only:
                continue
            m, color, offset = markers[name]
            ax.scatter(c - dx, r - dy, s=110, marker=m, facecolors="none", edgecolors=color, linewidths=2)
            ax.annotate(name, (c - dx, r - dy), xytext=offset, textcoords="offset points",
                        fontsize=9, color=color, fontweight="bold")

    maps = [("ground truth", truth)] + [(label, res["expected"]) for label, res in ckpt_results]
    for c, (label, field) in enumerate(maps):
        ax = fig.add_subplot(grid[0, c])
        ax.imshow(np.ma.masked_invalid(field), cmap=dbz_cmap, norm=dbz_norm, origin="lower",
                  interpolation="nearest")
        mark(ax)
        ax.set_title(label if c == 0 else f"{label}: expected dBZ", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    for p, (name, (r, c)) in enumerate(points.items(), start=1):
        gt = float(truth[r, c])
        # GT zoom, pixel coordinates kept so the marker lands on the point.
        r0, r1 = max(r - zoom_px, 0), min(r + zoom_px + 1, truth.shape[0])
        c0, c1 = max(c - zoom_px, 0), min(c + zoom_px + 1, truth.shape[1])
        ax = fig.add_subplot(grid[p, 0])
        ax.imshow(np.ma.masked_invalid(truth[r0:r1, c0:c1]), cmap=dbz_cmap, norm=dbz_norm,
                  origin="lower", interpolation="nearest")
        mark(ax, only=name, dx=c0, dy=r0)
        ax.set_title(f"{name}: GT {gt:.1f} dBZ  (row {r}, col {c}, ±{zoom_px * 2} km)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        gt_bin = bucket_of(gt)
        for k, (label, res) in enumerate(ckpt_results, start=1):
            probs = bin_probabilities(res["exceed_hist"][:, r, c])
            ax = fig.add_subplot(grid[p, k])
            bars = ax.bar(range(len(probs)), probs, color=bar_colors, edgecolor="0.4", linewidth=0.5)
            bars[gt_bin].set_edgecolor("red")
            bars[gt_bin].set_linewidth(2.5)
            # Shade the GT bucket too: when the model gives it ~0 probability
            # (the interesting case) its bar has no height to outline.
            ax.axvspan(gt_bin - 0.5, gt_bin + 0.5, color="red", alpha=0.12, zorder=0)
            ax.set_xticks(range(len(probs)), labels, rotation=60, fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_ylabel("predicted probability", fontsize=9)
            ax.set_xlabel("dBZ bucket (red = GT bucket)", fontsize=8)
            ax.set_title(f"{label} @ {name}", fontsize=10)
            e_k = res["exceed_hist"][:, r, c]
            info = [
                ("GT", f"{gt:.1f} dBZ"),
                ("expected", f"{res['expected'][r, c]:.1f} dBZ"),
                ("0.5 fcst", f"{res['forecast'][r, c]:.1f} dBZ"),
                ("P(>=20)", f"{e_k[BUCKET_EDGES.index(20.0)]:.2f}"),
                ("P(>=34)", f"{e_k[BUCKET_EDGES.index(34.0)]:.2f}"),
            ]
            ax.text(0.98, 0.97, "\n".join(f"{k:<9}{v:>9}" for k, v in info),
                    transform=ax.transAxes, ha="right", va="top", fontsize=8, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.9))

    fig.suptitle(f"RainPro-8 TW  |  predicted distribution at {lead_label(lead)}  |  "
                 f"init {np.datetime_as_string(init_time, unit='m')}", fontsize=13)
    save(fig, out_path)


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
                     help="comma-separated minutes, multiples of 10 in 10..360 (one figure column each)")
    ap.add_argument("--prob-thresholds", default="20,34",
                     help="comma-separated dBZ for the P(>t) rows; must be bucket boundaries "
                          f"{BUCKET_EDGES}")
    ap.add_argument("--mass-thresholds", default="20,34,40",
                     help="comma-separated dBZ (bucket boundaries) for the probability-mass table")
    ap.add_argument("--hist-lead", type=int, default=10,
                     help="lead time (min) whose GT picks the histogram grid points")
    ap.add_argument("--min-point", default="echo", choices=("echo", "any"),
                     help="GT-min point over echo pixels only (GT > 0, default) or over every "
                          "covered pixel (then usually a 0 dBZ pixel near the storm)")
    ap.add_argument("--out", default="inference.png",
                     help="output path stem: <stem>_overview.png, <stem>_hist.png, <stem>_<ckpt>.png")
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
    if args.hist_lead not in target_spec.offsets_min:
        raise SystemExit(f"--hist-lead {args.hist_lead} not in the target's offsets")
    hist_idx = list(target_spec.offsets_min).index(args.hist_lead)

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
    # Checkpoint-independent, so computed once. Built from the datamodule's
    # sources/norm_bounds -- the same ones the batch above was read with.
    optical_flow = OpticalFlowBaseline.from_sources(datamodule.sources, datamodule.norm_bounds)
    flow = optical_flow(batch)[0, lead_idx, 0].cpu().numpy()  # (n_lead, H, W) dBZ
    truth_hist = batch["target_2km"][0, hist_idx, 0].cpu().numpy()
    points = pick_points(truth_hist, args.min_point)
    print(f"histogram points at {lead_label(args.hist_lead)}: "
          + ", ".join(f"{n} (row {r}, col {c}) = {truth_hist[r, c]:.1f} dBZ" for n, (r, c) in points.items()))

    stem, ext = os.path.splitext(args.out)
    ext = ext or ".png"
    init_str = np.datetime_as_string(init_time, unit="m")
    ckpt_results: list[tuple[str, dict]] = []
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

        ckpt_results.append((ckpt_label, {
            "expected": expected,  # (n_lead, H, W)
            "expected_hist": expected_dbz(out.probs)[0, hist_idx].cpu().numpy(),
            "forecast_hist": out.forecast[0, hist_idx, 0].cpu().numpy(),
            "exceed_hist": (1.0 - out.probs[0, hist_idx]).cpu().numpy(),  # (K, H, W)
        }))

        rows: list[tuple[str, np.ndarray, str]] = [
            ("ground truth", truth, "dbz"),
            ("persistence (t0)", persistence, "dbz"),
            ("optical flow", flow, "dbz"),
            ("expected dBZ", expected, "dbz"),
            ("0.5 forecast", forecast, "dbz"),
            *[(f"P(>{t:g} dBZ)", exceed[:, k], "prob") for t, k in zip(prob_thresholds, prob_idx)],
        ]
        plot_grid(f"{stem}_{ckpt_label}{ext}", f"RainPro-8 TW  |  {ckpt_label}  |  init {init_str}",
                  lead_times, rows)

        header = "  lead    " + "  ".join(f"CSI{t:g} fcst/pers/flow".rjust(19) for t in prob_thresholds)
        print(header)
        for row, lt in enumerate(lead_times):
            cells = [
                "/".join(f"{csi(f[row], truth[row], t):.3f}" for f in (forecast, persistence, flow)).rjust(19)
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

    plot_grid(
        f"{stem}_overview{ext}",
        f"RainPro-8 TW  |  init {init_str}  |  expected dBZ from ordinal probabilities",
        lead_times,
        [("ground truth", truth, "dbz"), ("optical flow", flow, "dbz")]
        + [(label, res["expected"], "dbz") for label, res in ckpt_results],
    )
    plot_hist(
        f"{stem}_hist{ext}",
        init_time,
        args.hist_lead,
        truth_hist,
        points,
        [(label, {"expected": res["expected_hist"], "forecast": res["forecast_hist"],
                  "exceed_hist": res["exceed_hist"]}) for label, res in ckpt_results],
    )


if __name__ == "__main__":
    main()
