"""How much echo the training targets actually contain -- and how much
training signal each ordinal channel of the loss gets from it.

Reads the target exactly as `RainPro8Dataset` builds `target_2km` (same splits
after `RainPro8DataModule.setup()`'s STA_H8 filtering, same 2 km canvas, -99 ->
0 dBZ, -999 -> NaN, no jitter), but each QPESUMS frame only once: consecutive
init times share 35 of their 36 target frames, so per-frame statistics are
computed once and each sample's are assembled by index.

Reports, per split:

  1. Pixel level (what the loss sees): fraction of valid target pixels >= each
     bucket edge, and per ordinal channel c the conditional task it trains --
     supervised pixels (target >= t_{c-1}) and their positive rate
     P(>= t_c | >= t_{c-1}).
  2. Sample level: how many samples (36 target frames each) have no echo at
     all, and how the echo coverage (fraction of pixels >= 20 / >= 35 dBZ) is
     distributed; fraction of samples with any pixel >= 35 / 40 / 45 dBZ.
  3. Batch level: with shuffled batches of `--batch-size`, the chance a
     micro-batch has NO pixel >= t -- `OrdinalConsistentLoss` then skips that
     channel entirely (all-NaN), i.e. it gets no gradient from that step.
  4. By month: sample count and mean coverage, to see where the echo lives.

Plus a clutter check over every frame read (all requested splits), since
strong echo showed up in literally every sample, dry season included:
per-pixel frequency of >= 35 / >= 45 dBZ, and of >= 35 dBZ in *dry* frames
(canvas fraction >= 20 dBZ below `--dry-frac`). Rain doesn't sit on one
pixel through dry weather; ground/sea clutter and anomalous propagation do.
Lists the top hotspots (ranked by dry-frame frequency, with their frequency
relative to the 9x9 neighbourhood median -- clutter is isolated, rain is
spatially smooth), how many pixels exceed a few dry-frequency levels, and what
share of all >= 35 dBZ training pixels they account for. Writes
`<out stem>_clutter.png` and, with `--clutter-npz`, the raw maps (plus the
target grid's lat/lon) for building a static mask later.

">=" matches the loss's `Bucketize(right=True)` (a target of exactly t counts
as exceeding t). Writes a per-sample CSV and a summary figure.

Presentation figures (audience-facing, `--lang zh` by default), counted per
10-minute frame -- each frame once, not repeated across the 36 overlapping
samples that contain it:
  <out stem>_coverage.png      how much of the canvas reaches >= 20 / >= 35 dBZ,
                               across all training frames, in readable bins.
  <out stem>_mean_dbz.png      histogram of each training frame's canvas-mean
                               reflectivity (no echo counted as 0 dBZ).
  <out stem>_train_vs_val.png  share of grid cells >= 20 / 35 / 45 dBZ per
                               split, and >= 35 dBZ coverage by month (needs
                               val in --splits; test is drawn too when present).
Titles only name what is plotted; they state no findings.

Usage (CPU, no checkpoint; reads every 2021 QPESUMS frame once):
    python scripts/inspect_echo_stats.py --config rainpro8.yml \\
        --data-root '{"qpesums": "...", "sta_h8": "...Q1.zarr,...Q2.zarr,..."}' \\
        --variable-aliases '{"max_dbz": "MaxDBZ"}' \\
        --splits train,val,test --out figures/echo_stats.png --csv figures/echo_stats.csv \\
        --clutter-npz figures/clutter_maps.npz
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import median_filter

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainpro.data.rainpro8_datamodule import RainPro8DataModule  # noqa: E402
from rainpro.data.rainpro8_dataset import (  # noqa: E402
    RainPro8Dataset,
    _fill_no_echo,
    _mask_missing,
    _tolerance,
)
from rainpro.data.regrid import target_grid  # noqa: E402
from rainpro.loss.ordinal_consistent import taiwan_dbz_buckets  # noqa: E402

from plot_bucket_distribution import setup_font  # noqa: E402  (same scripts/ directory)

EDGES = [b.min for b in taiwan_dbz_buckets()]
# Report thresholds; must be a subset of COUNT_AT (which is what gets counted).
REPORT = (5.0, 20.0, 30.0, 35.0, 40.0, 45.0)
COUNT_AT = sorted(set(EDGES) | set(REPORT))
COVERAGE_BINS = [0, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 1.0]  # fraction of sample pixels


def load_datamodule(args) -> RainPro8DataModule:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["data"]
    cfg = cfg.get("init_args", cfg)
    accepted = set(inspect.signature(RainPro8DataModule.__init__).parameters) - {"self"}
    cfg = {k: v for k, v in cfg.items() if k in accepted}
    cfg["data_root"] = json.loads(args.data_root)
    if args.variable_aliases:
        cfg["variable_aliases"] = json.loads(args.variable_aliases)
    return RainPro8DataModule(**cfg)


def frame_stats(dm: RainPro8DataModule, positions: np.ndarray, dry_frac: float, block: int = 64):
    """Per QPESUMS time position: (valid pixel count, counts >= COUNT_AT, max)
    on the target_2km canvas, plus per-pixel count maps accumulated over all
    of `positions` for the clutter check (see module docstring)."""
    spec = dm.sources["target_2km"]
    ds = RainPro8Dataset(dm.data_root, {"target_2km": spec}, [], center_lat=dm.center_lat,
                         center_lon=dm.center_lon, variable_aliases=dm.variable_aliases,
                         latlon_names=dm.latlon_names)
    handle = ds._get_store("qpesums")
    regridder = ds._get_regridder("qpesums")
    mapping = regridder.prepare(*target_grid(dm.center_lat, dm.center_lon, spec.size_km, spec.resolution_km))
    raw_name = (dm.variable_aliases or {}).get(spec.variables[0], spec.variables[0])
    thresholds = np.asarray(COUNT_AT, dtype=np.float32)

    valid = np.zeros(len(positions), dtype=np.int64)
    counts = np.zeros((len(positions), len(COUNT_AT)), dtype=np.int64)
    vmax = np.full(len(positions), np.nan, dtype=np.float32)
    vmean = np.full(len(positions), np.nan, dtype=np.float32)  # grid mean, no echo counted as 0 dBZ
    maps = {k: np.zeros(mapping.dst_shape, dtype=np.int64)
            for k in ("valid", "ge35", "ge45", "dry_valid", "dry_ge35")}
    n_dry = 0
    for i in range(0, len(positions), block):
        frames = handle.read(raw_name, positions[i:i + block])
        for j, frame in enumerate(frames):
            frame = _fill_no_echo(frame, spec.no_echo_values, spec.no_echo_fill)
            frame = _mask_missing(frame, spec.missing_values)
            field = regridder.apply(frame, mapping, fill_value=np.nan)
            finite = np.isfinite(field)
            vals = field[finite]
            valid[i + j] = vals.size
            if not vals.size:
                continue
            counts[i + j] = (vals[:, None] >= thresholds[None, :]).sum(axis=0)
            vmax[i + j] = vals.max()
            vmean[i + j] = vals.mean()
            with np.errstate(invalid="ignore"):  # NaN >= t is False, as wanted
                ge35, ge45 = field >= 35.0, field >= 45.0
            maps["valid"] += finite
            maps["ge35"] += ge35
            maps["ge45"] += ge45
            if (vals >= 20.0).mean() < dry_frac:
                n_dry += 1
                maps["dry_valid"] += finite
                maps["dry_ge35"] += ge35
        print(f"  frames {min(i + block, len(positions))}/{len(positions)}", flush=True)
    maps["n_frames"], maps["n_dry"] = len(positions), n_dry
    maps["lat"], maps["lon"] = target_grid(dm.center_lat, dm.center_lon, spec.size_km, spec.resolution_km)
    return valid, counts, vmax, vmean, maps


def clutter_report(maps: dict, out_png: str, top_n: int, npz: str | None) -> None:
    """Hotspots of strong echo that persist in dry weather. See module docstring."""
    with np.errstate(invalid="ignore", divide="ignore"):
        f35 = maps["ge35"] / maps["valid"]
        f45 = maps["ge45"] / maps["valid"]
        dry35 = maps["dry_ge35"] / maps["dry_valid"]
    f35, f45, dry35 = (np.nan_to_num(a) for a in (f35, f45, dry35))
    neighbourhood = median_filter(f35, size=9, mode="nearest")
    isolation = f35 / (neighbourhood + 1e-4)

    print(f"\n==================== clutter check ====================")
    print(f"frames read: {maps['n_frames']}, dry frames (canvas >= 20 dBZ below --dry-frac): {maps['n_dry']}")
    if maps["n_dry"] == 0:
        print("   no dry frames -- lower-frequency hotspots can't be separated from rain; raise --dry-frac")
    total35 = maps["ge35"].sum()
    for level in (0.01, 0.05, 0.2):
        cand = dry35 >= level
        share = maps["ge35"][cand].sum() / total35 if total35 else float("nan")
        print(f"   pixels with dry-frame freq(>=35) >= {level:.0%}: {cand.sum():5d}"
              f"  -> {share:.1%} of all >= 35 dBZ target pixels")

    order = np.argsort(dry35.ravel())[::-1][:top_n]
    order = order[dry35.ravel()[order] > 0]  # never list / circle pixels that never fired
    print(f"   top {len(order)} by dry-frame freq(>=35):")
    print(f"   {'row':>4} {'col':>4} {'lat':>7} {'lon':>8} {'f>=35':>7} {'f>=45':>7} {'dry f>=35':>9}"
          f" {'nbhd med':>8} {'isolation':>9}")
    for flat in order:
        r, c = np.unravel_index(flat, f35.shape)
        print(f"   {r:>4} {c:>4} {maps['lat'][r, c]:>7.3f} {maps['lon'][r, c]:>8.3f} {f35[r, c]:>7.3f}"
              f" {f45[r, c]:>7.3f} {dry35[r, c]:>9.3f} {neighbourhood[r, c]:>8.3f} {isolation[r, c]:>9.1f}")

    if npz:
        os.makedirs(os.path.dirname(os.path.abspath(npz)), exist_ok=True)
        np.savez_compressed(npz, **{k: np.asarray(v) for k, v in maps.items()})
        print(f"wrote {npz}")

    panels = [
        ("freq >= 35 dBZ (all frames)", f35, mpl.colors.LogNorm(1e-4, 1)),
        ("freq >= 45 dBZ (all frames)", f45, mpl.colors.LogNorm(1e-4, 1)),
        (f"freq >= 35 dBZ in dry frames (n={maps['n_dry']})", dry35, mpl.colors.LogNorm(1e-4, 1)),
        ("isolation: freq>=35 / 9x9 median", isolation, mpl.colors.LogNorm(1, max(10.0, float(isolation.max())))),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), layout="constrained")
    rr, cc = np.unravel_index(order, f35.shape)
    for ax, (title, field, norm) in zip(axes, panels):
        cmap = mpl.colormaps["magma_r"].copy()
        cmap.set_bad("white")
        im = ax.imshow(np.ma.masked_less_equal(field, 0), cmap=cmap, norm=norm, origin="lower",
                       interpolation="nearest")
        ax.scatter(cc, rr, s=60, facecolors="none", edgecolors="cyan", linewidths=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"clutter check on the target_2km canvas  |  cyan: top {top_n} by dry-frame frequency",
                 fontsize=12)
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


# --- presentation figures ---------------------------------------------------
# Reference palette (light): categorical slots 1-3 validated all-pairs; aqua is
# below 3:1 on the surface, so every bar carries a direct value label.
SPLIT_COLOR = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COVER_BINS = [(0, 0), (0, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 5e-2), (5e-2, 0.2), (0.2, 1.0 + 1e-9)]
TEXT = {
    "zh": {
        "split": {"train": "訓練集", "val": "驗證集", "test": "測試集"},
        "cover_labels": ["0", "<0.01%", "0.01–0.1%", "0.1–1%", "1–5%", "5–20%", "≥20%"],
        "cover_x": "回波覆蓋率（畫面中達到門檻的網格比例）",
        "cover_y": "frame 比例（%）",
        "cover_title": "≥{t:g} dBZ",
        "cover_sup": "訓練資料的回波覆蓋率分布",
        "frames_note": "訓練集 {n:,} 個 frame（每 10 分鐘一張，512 × 512 km）",
        "mean_x": "frame 的全畫面平均回波（dBZ，無回波格點以 0 計）",
        "mean_y": "frame 比例（%）",
        "mean_title": "訓練資料每個 frame 的全畫面平均回波",
        "median": "中位數",
        "share_title": "≥{t:g} dBZ",
        "share_y": "達到門檻的網格比例（%）",
        "month_title": "逐月 ≥35 dBZ 覆蓋率",
        "month_x": "月份",
        "month_y": "平均覆蓋率（%）",
        "vs_sup": "訓練集、驗證集與測試集的回波分布",
        "vs_note": "frame 數：{counts}（每 10 分鐘一張；比例以網格計）",
    },
    "en": {
        "split": {"train": "train", "val": "validation", "test": "test"},
        "cover_labels": ["0", "<0.01%", "0.01–0.1%", "0.1–1%", "1–5%", "5–20%", "≥20%"],
        "cover_x": "echo coverage (share of the canvas at or above the threshold)",
        "cover_y": "share of frames (%)",
        "cover_title": "≥{t:g} dBZ",
        "cover_sup": "Echo coverage of the training frames",
        "frames_note": "{n:,} training frames (one every 10 min, 512 × 512 km)",
        "mean_x": "canvas-mean reflectivity of a frame (dBZ, no echo counted as 0)",
        "mean_y": "share of frames (%)",
        "mean_title": "Canvas-mean reflectivity per training frame",
        "median": "median",
        "share_title": "≥{t:g} dBZ",
        "share_y": "share of grid cells (%)",
        "month_title": "Monthly coverage ≥35 dBZ",
        "month_x": "month",
        "month_y": "mean coverage (%)",
        "vs_sup": "Echo distribution of the train, validation and test splits",
        "vs_note": "frames: {counts} (one every 10 min; shares are per grid cell)",
    },
}


def _style(ax, grid_axis="y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=11, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar_labels(ax, bars, values, fmt) -> None:
    """Value on top of each bar; empty bars stay unlabeled (a row of '0.0%'
    is noise, and the missing bar already says zero)."""
    for bar, v in zip(bars, values):
        if v == 0:
            continue
        ax.annotate(fmt(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=11, color=INK_2)


def presentation_figures(stem: str, ext: str, lang: str, frames: dict, frame_times: pd.DatetimeIndex) -> None:
    """Three audience-facing figures. `frames[split]` = indices into the
    per-frame arrays (f20, f35, f45, mean_dbz, valid, ge20/35/45 counts) of
    the frames that split's targets use; each frame counted once."""
    text = TEXT[lang]
    fig_kw = dict(layout="constrained", facecolor=SURFACE)

    # 1. coverage distribution of training frames
    tr = frames["train"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True, **fig_kw)
    for ax, t in zip(axes, (20.0, 35.0)):
        cover = tr[f"f{t:g}"]
        shares = [np.mean(cover == 0) if hi == 0 else np.mean((cover > 0) & (cover >= lo) & (cover < hi))
                  for lo, hi in COVER_BINS]
        shares = [100 * x for x in shares]
        bars = ax.bar(range(len(shares)), shares, width=0.72, color=SPLIT_COLOR["train"],
                      edgecolor=SURFACE, linewidth=2)
        _bar_labels(ax, bars, shares, lambda v: f"{v:.1f}%")
        _style(ax)
        ax.set_xticks(range(len(shares)), text["cover_labels"])
        ax.set_xlabel(text["cover_x"], fontsize=12, color=INK_2)
        ax.set_title(text["cover_title"].format(t=t),
                     fontsize=14, color=INK, loc="left")
    axes[0].set_ylabel(text["cover_y"], fontsize=12, color=INK_2)
    axes[0].set_ylim(0, max(b.get_height() for ax in axes for b in ax.patches) * 1.15)
    fig.suptitle(text["cover_sup"], fontsize=16, color=INK, x=0.01, ha="left")
    fig.supxlabel(text["frames_note"].format(n=len(tr["f20"])), fontsize=10, color=MUTED, x=0.99, ha="right")
    _save(fig, f"{stem}_coverage{ext}")

    # 2. canvas-mean dBZ per frame
    mean = tr["mean_dbz"][np.isfinite(tr["mean_dbz"])]
    fig, ax = plt.subplots(figsize=(11, 5), **fig_kw)
    # ~30 bins over the observed range, on a round step so bin edges read cleanly
    step = next(w for w in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0) if mean.max() / w <= 40)
    edges = np.arange(0, np.ceil(mean.max() / step) * step + step, step)
    counts_, _ = np.histogram(mean, bins=edges)
    heights = 100 * counts_ / len(mean)
    ax.bar(edges[:-1], heights, width=step, align="edge", color=SPLIT_COLOR["train"],
           edgecolor=SURFACE, linewidth=1.5)
    med = float(np.median(mean))
    top = heights.max() * 1.18
    ax.set_ylim(0, top)
    ax.axvline(med, color=INK_2, linewidth=1.5, linestyle="--")
    ax.annotate(f"{text['median']} {med:.1f} dBZ", (med, top), xytext=(6, -4), textcoords="offset points",
                fontsize=11, color=INK, va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none"))
    _style(ax)
    ax.set_xlabel(text["mean_x"], fontsize=12, color=INK_2)
    ax.set_ylabel(text["mean_y"], fontsize=12, color=INK_2)
    ax.set_title(text["mean_title"], fontsize=15, color=INK, loc="left")
    fig.supxlabel(text["frames_note"].format(n=len(mean)), fontsize=10, color=MUTED, x=0.99, ha="right")
    _save(fig, f"{stem}_mean_dbz{ext}")

    # 3. train vs val (vs test)
    if "val" not in frames:
        print("!! no val split requested -- skipping the train-vs-val figure (add val to --splits)")
        return
    splits = [s for s in ("train", "val", "test") if s in frames]
    share = {s: {t: frames[s][f"ge{t:g}"].sum() / frames[s]["valid"].sum() for t in (20.0, 35.0, 45.0)}
             for s in splits}
    fig = plt.figure(figsize=(14, 9), **fig_kw)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    for c, t in enumerate((20.0, 35.0, 45.0)):
        ax = fig.add_subplot(grid[0, c])
        vals = [100 * share[s][t] for s in splits]
        bars = ax.bar(range(len(splits)), vals, width=0.62, color=[SPLIT_COLOR[s] for s in splits],
                      edgecolor=SURFACE, linewidth=2)
        _bar_labels(ax, bars, vals, lambda v: f"{v:.2g}%")
        _style(ax)
        ax.set_xticks(range(len(splits)), [text["split"][s] for s in splits], fontsize=12)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(text["share_title"].format(t=t), fontsize=14, color=INK, loc="left")
        if c == 0:
            ax.set_ylabel(text["share_y"], fontsize=12, color=INK_2)

    ax = fig.add_subplot(grid[1, :])
    monthly = {}
    for s in splits:
        months = frame_times[frames[s]["index"]].month
        monthly[s] = pd.Series(frames[s]["f35"]).groupby(months).mean() * 100
        ax.plot(monthly[s].index, monthly[s].values, color=SPLIT_COLOR[s], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=text["split"][s])
    _style(ax)
    ax.set_xticks(range(1, 13))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, None)
    ax.set_xlabel(text["month_x"], fontsize=12, color=INK_2)
    ax.set_ylabel(text["month_y"], fontsize=12, color=INK_2)
    ax.set_title(text["month_title"], fontsize=14, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    fig.suptitle(text["vs_sup"], fontsize=16, color=INK, x=0.01, ha="left")
    counts_txt = ", ".join(f"{text['split'][s]} {len(frames[s]['f35']):,}" for s in splits)
    fig.supxlabel(text["vs_note"].format(counts=counts_txt), fontsize=10, color=MUTED, x=0.99, ha="right")
    _save(fig, f"{stem}_train_vs_val{ext}")


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="rainpro8.yml or a run's saved config.yml")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--variable-aliases", default=None)
    ap.add_argument("--splits", default="train", help="comma-separated: train,val,test")
    ap.add_argument("--batch-size", type=int, default=4,
                     help="micro-batch size for the no-gradient estimate (your sbatch: 4 x accumulate 4)")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th init time (faster, coarser)")
    ap.add_argument("--out", default="echo_stats.png")
    ap.add_argument("--csv", default=None, help="optional per-sample CSV")
    ap.add_argument("--dry-frac", type=float, default=1e-3,
                     help="a frame is 'dry' when this fraction of the canvas or less is >= 20 dBZ")
    ap.add_argument("--top-n", type=int, default=20, help="hotspots to list / circle")
    ap.add_argument("--clutter-npz", default=None, help="optional: save the per-pixel maps")
    ap.add_argument("--lang", default="zh", choices=("zh", "en"), help="labels of the presentation figures")
    ap.add_argument("--font", default=None, help="font file with CJK glyphs, if none is installed")
    args = ap.parse_args()

    dm = load_datamodule(args)
    dm.setup()
    spec = dm.sources["target_2km"]
    offsets = np.asarray(spec.offsets_min)
    splits = [s.strip() for s in args.splits.split(",")]

    # Resolve every sample's 36 target frames to store positions, like the Dataset.
    sample_times, sample_pos = {}, {}
    probe = RainPro8Dataset(dm.data_root, {"target_2km": spec}, [], variable_aliases=dm.variable_aliases,
                            latlon_names=dm.latlon_names)
    time_index = probe._get_store("qpesums").time_index
    for split in splits:
        times = pd.DatetimeIndex(dm.split_times[split])[:: args.stride]
        query = (times.values[:, None] + offsets[None, :].astype("timedelta64[m]")).ravel()
        pos = time_index.get_indexer(pd.DatetimeIndex(query), method="nearest", tolerance=_tolerance(spec))
        sample_times[split], sample_pos[split] = times, pos.reshape(len(times), len(offsets))
        print(f"{split}: {len(times)} samples", flush=True)

    unique = np.unique(np.concatenate([p[p >= 0] for p in sample_pos.values()]))
    print(f"reading {len(unique)} unique QPESUMS frames ...", flush=True)
    valid_u, counts_u, max_u, mean_u, maps = frame_stats(dm, unique, args.dry_frac)

    rep_idx = [COUNT_AT.index(t) for t in REPORT]
    edge_idx = [COUNT_AT.index(t) for t in EDGES]
    csv_rows, results = [], {}
    for split in splits:
        pos = sample_pos[split]
        ok = pos >= 0  # (N, 36); False = no frame within tolerance (stays NaN in the Dataset)
        safe = np.where(ok, np.searchsorted(unique, pos), 0)  # row into the per-frame arrays
        valid = np.where(ok, valid_u[safe], 0).sum(axis=1)  # (N,)
        counts = np.where(ok[..., None], counts_u[safe], 0).sum(axis=1)  # (N, len(COUNT_AT))
        smax = np.where(ok, np.nan_to_num(max_u, nan=-np.inf)[safe], -np.inf).max(axis=1)
        smax[np.isneginf(smax)] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = counts / valid[:, None]
        results[split] = dict(valid=valid, counts=counts, frac=frac, smax=smax, times=sample_times[split])

        n = len(valid)
        tot_valid = valid.sum()
        tot = counts.sum(axis=0)
        print(f"\n==================== {split}: {n} samples ====================")
        print("1) pixel level -- fraction of valid target pixels >= t")
        print("   " + "  ".join(f">={t:g}".rjust(9) for t in REPORT))
        print("   " + "  ".join(f"{tot[k] / tot_valid:9.2e}" for k in rep_idx))

        print("   ordinal channels (what each conditional sigmoid is trained on):")
        print(f"   {'ch':>3} {'edge':>5} {'supervised px':>14} {'share of all':>12} {'P(>=t_c | >=t_c-1)':>19}")
        prev = tot_valid
        for c, k in enumerate(edge_idx):
            pos_c = tot[k]
            print(f"   {c:>3} {EDGES[c]:>5g} {prev:>14,d} {prev / tot_valid:>12.2e} {pos_c / prev if prev else float('nan'):>19.3f}")
            prev = pos_c

        print("2) sample level (36 target frames per sample)")
        no_echo = (counts[:, COUNT_AT.index(5.0)] == 0).mean()
        print(f"   no pixel >= 5 dBZ in any frame: {no_echo:.1%}")
        for t in (20.0, 35.0):
            f = frac[:, COUNT_AT.index(t)]
            hist, _ = np.histogram(np.nan_to_num(f), bins=COVERAGE_BINS)
            print(f"   coverage >= {t:g} dBZ: " + ", ".join(
                f"[{a:g},{b:g}) {h / n:.1%}" for a, b, h in zip(COVERAGE_BINS[:-1], COVERAGE_BINS[1:], hist)))
        for t in (35.0, 40.0, 45.0):
            print(f"   samples with any pixel >= {t:g} dBZ: {(counts[:, COUNT_AT.index(t)] > 0).mean():.1%}")

        print(f"3) batch level (batch {args.batch_size}, shuffled): P(no pixel >= t in the batch)"
              " -> that channel gets no gradient that step")
        for t in (20.0, 34.0, 40.0, 46.0, 52.0):
            p_sample = (counts[:, COUNT_AT.index(t)] > 0).mean()
            print(f"   >= {t:g} dBZ: sample has it {p_sample:.1%}, batch lacks it {(1 - p_sample) ** args.batch_size:.1%}")

        print("4) by month")
        months = pd.DatetimeIndex(sample_times[split]).month
        print(f"   {'month':>5} {'samples':>8} {'mean frac>=20':>14} {'mean frac>=35':>14} {'any>=35':>8}")
        for m in sorted(set(months)):
            sel = months == m
            print(f"   {m:>5} {sel.sum():>8} {np.nanmean(frac[sel, COUNT_AT.index(20.0)]):>14.2e}"
                  f" {np.nanmean(frac[sel, COUNT_AT.index(35.0)]):>14.2e}"
                  f" {(counts[sel, COUNT_AT.index(35.0)] > 0).mean():>8.1%}")

        for i, t0 in enumerate(sample_times[split]):
            csv_rows.append({"split": split, "init_time": str(t0), "valid_px": int(valid[i]),
                             "max_dbz": float(smax[i]),
                             **{f"frac_ge{t:g}": float(frac[i, COUNT_AT.index(t)]) for t in REPORT}})

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nwrote {args.csv}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
    colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    for split, res in results.items():
        tot = res["counts"].sum(axis=0) / res["valid"].sum()
        axes[0].semilogy(EDGES, [tot[COUNT_AT.index(e)] for e in EDGES], "o-", color=colors.get(split), label=split)
        for t, ls in ((20.0, "-"), (35.0, "--")):
            f = res["frac"][:, COUNT_AT.index(t)]
            f = np.sort(np.nan_to_num(f))
            axes[1].plot(f, np.linspace(0, 1, len(f)), ls=ls, color=colors.get(split), label=f"{split} >={t:g}")
        months = pd.DatetimeIndex(res["times"]).month
        mf = [np.nanmean(res["frac"][months == m, COUNT_AT.index(35.0)]) if (months == m).any() else np.nan
              for m in range(1, 13)]
        axes[2].plot(range(1, 13), mf, "o-", color=colors.get(split), label=split)
    axes[0].set(xlabel="bucket edge t (dBZ)", ylabel="fraction of valid pixels >= t",
                title="pixel-level exceedance (what the loss sees)")
    axes[1].set(xscale="symlog", xlabel="fraction of a sample's pixels >= t (symlog)", ylabel="CDF over samples",
                title="per-sample echo coverage")
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[2].set(xlabel="month", ylabel="mean fraction >= 35 dBZ", title="strong-echo coverage by month",
                xticks=range(1, 13))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    stem, ext = os.path.splitext(args.out)
    ext = ext or ".png"
    clutter_report(maps, f"{stem}_clutter{ext}", args.top_n, args.clutter_npz)

    # Presentation figures work per FRAME (each 10-min frame once), not per
    # sample (where every frame is repeated across 36 overlapping samples).
    frames = {}
    for split in splits:
        pos = sample_pos[split]
        idx = np.unique(np.searchsorted(unique, pos[pos >= 0]))
        idx = idx[valid_u[idx] > 0]
        frames[split] = {"index": idx, "valid": valid_u[idx], "mean_dbz": mean_u[idx]}
        for t in (20.0, 35.0, 45.0):
            frames[split][f"ge{t:g}"] = counts_u[idx, COUNT_AT.index(t)]
            frames[split][f"f{t:g}"] = counts_u[idx, COUNT_AT.index(t)] / valid_u[idx]
    if "train" in frames:
        lang = setup_font(args.lang, args.font)
        presentation_figures(stem, ext, lang, frames, time_index[unique])
    else:
        print("!! presentation figures need the train split in --splits")


if __name__ == "__main__":
    main()
