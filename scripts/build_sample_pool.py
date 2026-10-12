"""Build the sample pool for cross-validation: every usable init time, its echo
statistics, and whether the low-echo filter keeps it for training.

Pool = every 10-minute init time in [start_date, end_date) that passes the
same availability checks training uses (`RainPro8DataModule.setup()`: QPESUMS
present, all 9 STA_H8 bands present at the satellite offsets), with no
train/val/test split applied -- the CV step decides that later.

Low-echo filter. A sample is "low echo" when its coverage >= 20 dBZ (share
of valid grid cells on the target_2km canvas) is below `--min-cov20`
(default 5%), measured on `--basis`:
  target  pooled over the 36 target frames (+10 .. +360 min), i.e. what the
          loss actually sees (default);
  t0      the radar frame at the init time only.
Low-echo samples are dropped from TRAINING, except a random `--keep-low-frac`
share of them (default 0 = drop all), kept with `weight` = 1 / keep_low_frac
so a later step can undo the base-rate shift if wanted. Dropping all of them
teaches the model that echo is far more common than it is; expect its
probabilities to run high on dry or light-rain days.

The filter is for the TRAINING folds only. Validation and test must be scored
on unfiltered samples, or the scores describe rainy days rather than the
forecast problem.

Writes `--out` (CSV, one row per pool sample):
  init_time (store time, UTC) / local_time / date_local / month / season /
  typhoon             calendar keys for grouping and stratifying folds
  cov20 / cov35 / cov45_target, cov20_t0
                      coverage >= 20 / 35 / 45 dBZ
  max_dbz_target      strongest target pixel
  n_target_frames     target frames found (36 = complete)
  invalid_dbz         some target frame exceeds `--max-valid-dbz` (data error)
  low_echo, keep, weight
and prints the threshold sensitivity, kept share by season / month, and the
invalid-frame count. With `--figure`, also a two-panel figure (kept vs dropped
per month, kept share vs threshold).

Seasons (local time; meiyu May 15 - Jun 30, then summer):
  winter Nov-Mar | spring Apr-May 14 | meiyu May 15-Jun 30 | summer Jul-Sep | autumn Oct
`typhoon` uses scripts/find_weather_cases.py's windows (`--typhoon` to
override; same caveat: from memory, check against CWA).

Usage (CPU; reads every QPESUMS frame of the period once, ~15 min for 2021):
    python scripts/build_sample_pool.py --config rainpro8.yml \\
        --data-root "$DATA_ROOT" --variable-aliases '{"max_dbz": "MaxDBZ"}' \\
        --out figures/sample_pool_2021.csv --figure figures/sample_pool_2021.png
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from find_weather_cases import DEFAULT_TYPHOONS  # noqa: E402  (same scripts/ directory)
from inspect_echo_stats import (  # noqa: E402
    COUNT_AT,
    INK,
    INK_2,
    MUTED,
    SPLIT_COLOR,
    SURFACE,
    _style,
    frame_stats,
)

from rainpro.data.rainpro8_datamodule import RainPro8DataModule  # noqa: E402
from rainpro.data.rainpro8_dataset import RainPro8Dataset, _tolerance  # noqa: E402

SENSITIVITY = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
SEASONS = ("winter", "spring", "meiyu", "summer", "autumn")
DROPPED_GREY = "#c3c2b7"


def pool_datamodule(args) -> RainPro8DataModule:
    """The configured data module, but with one cycle spanning the whole
    period (no val/test days, no blackout) so `setup()` returns every usable
    init time as "train"."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["data"]
    cfg = cfg.get("init_args", cfg)
    accepted = set(inspect.signature(RainPro8DataModule.__init__).parameters) - {"self"}
    cfg = {k: v for k, v in cfg.items() if k in accepted}
    cfg["data_root"] = json.loads(args.data_root)
    if args.variable_aliases:
        cfg["variable_aliases"] = json.loads(args.variable_aliases)
    span = (pd.Timestamp(cfg["end_date"]) - pd.Timestamp(cfg["start_date"])).days + 1
    cfg.update(cycle_train_days=span, cycle_val_days=0, cycle_test_days=0, cycle_blackout_hours=0)
    cfg["sample_manifest"] = None  # the pool is what a manifest is built FROM; ignore any configured one
    return RainPro8DataModule(**cfg)


def season_of(local: pd.DatetimeIndex) -> np.ndarray:
    md = local.month * 100 + local.day
    return np.select(
        [(local.month >= 11) | (local.month <= 3), md < 515, md <= 630, local.month <= 9],
        ["winter", "spring", "meiyu", "summer"],
        default="autumn",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--variable-aliases", default=None)
    ap.add_argument("--min-cov20", type=float, default=0.05,
                     help="low echo = coverage >= 20 dBZ below this share of grid cells")
    ap.add_argument("--basis", default="target", choices=("target", "t0"))
    ap.add_argument("--keep-low-frac", type=float, default=0.0,
                     help="random share of low-echo samples kept anyway (weight 1/frac)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-valid-dbz", type=float, default=75.0)
    ap.add_argument("--drop-invalid", action="store_true",
                     help="also drop samples whose target holds a value above --max-valid-dbz")
    ap.add_argument("--tz-offset-hours", type=float, default=8.0)
    ap.add_argument("--typhoon", action="append", default=None,
                     help=f"repeatable NAME:START:END; default {DEFAULT_TYPHOONS}")
    ap.add_argument("--out", default="sample_pool.csv")
    ap.add_argument("--figure", default=None, help="optional PNG: kept/dropped per month + threshold curve")
    args = ap.parse_args()
    if not 0.0 <= args.keep_low_frac <= 1.0:
        raise SystemExit("--keep-low-frac must be in [0, 1]")

    dm = pool_datamodule(args)
    dm.setup()
    spec = dm.sources["target_2km"]
    inits = pd.DatetimeIndex(dm.split_times["train"])
    print(f"pool: {len(inits):,} usable init times", flush=True)

    probe = RainPro8Dataset(dm.data_root, {"target_2km": spec}, [], variable_aliases=dm.variable_aliases,
                            latlon_names=dm.latlon_names)
    time_index = probe._get_store("qpesums").time_index
    offsets = np.concatenate([[0], np.asarray(spec.offsets_min)])  # column 0 = t0
    query = (inits.values[:, None] + offsets[None, :].astype("timedelta64[m]")).ravel()
    pos = time_index.get_indexer(pd.DatetimeIndex(query), method="nearest",
                                 tolerance=_tolerance(spec)).reshape(len(inits), len(offsets))
    unique = np.unique(pos[pos >= 0])
    print(f"reading {len(unique):,} QPESUMS frames ...", flush=True)
    valid_u, counts_u, max_u, _, _ = frame_stats(dm, unique, dry_frac=1e-3)

    ok = pos >= 0
    row = np.where(ok, np.searchsorted(unique, pos), 0)
    k = {t: COUNT_AT.index(t) for t in (20.0, 35.0, 45.0)}

    def pooled(cols: slice, t: float) -> np.ndarray:
        okc, rowc = ok[:, cols], row[:, cols]
        hits = np.where(okc, counts_u[rowc, k[t]], 0).sum(axis=1)
        n = np.where(okc, valid_u[rowc], 0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, hits / n, np.nan)

    target, t0 = slice(1, None), slice(0, 1)
    local = inits + pd.Timedelta(hours=args.tz_offset_hours)
    pool = pd.DataFrame({
        "init_time": inits,
        "local_time": local,
        "date_local": local.normalize().date,
        "month": local.month,
        "season": season_of(local),
        "cov20_target": pooled(target, 20.0),
        "cov35_target": pooled(target, 35.0),
        "cov45_target": pooled(target, 45.0),
        "cov20_t0": pooled(t0, 20.0),
        "max_dbz_target": np.where(ok[:, target], np.nan_to_num(max_u, nan=-np.inf)[row[:, target]],
                                   -np.inf).max(axis=1),
        "n_target_frames": ok[:, target].sum(axis=1),
    })
    pool.loc[np.isneginf(pool["max_dbz_target"]), "max_dbz_target"] = np.nan
    pool["invalid_dbz"] = pool["max_dbz_target"] > args.max_valid_dbz

    pool["typhoon"] = ""
    for item in args.typhoon or DEFAULT_TYPHOONS:
        name, start, end = item.split(":")
        inside = (local >= pd.Timestamp(start)) & (local < pd.Timestamp(end) + pd.Timedelta(days=1))
        pool.loc[inside, "typhoon"] = name

    basis = pool["cov20_target"] if args.basis == "target" else pool["cov20_t0"]
    pool["low_echo"] = (basis < args.min_cov20) | basis.isna()
    # Per-init-time draw from a seed, so the kept subset is reproducible and
    # doesn't change when the pool's period or row order does.
    seed_draw = np.array([np.random.default_rng((args.seed, int(t.value // 60_000_000_000))).random()
                          for t in inits])
    rescued = pool["low_echo"] & (seed_draw < args.keep_low_frac)
    pool["keep"] = ~pool["low_echo"] | rescued
    if args.drop_invalid:
        pool.loc[pool["invalid_dbz"], "keep"] = False
    pool["weight"] = np.where(rescued, 1.0 / args.keep_low_frac if args.keep_low_frac else 0.0, 1.0)
    pool.loc[~pool["keep"], "weight"] = 0.0

    # --- report ------------------------------------------------------------
    n = len(pool)
    print(f"\nlow-echo rule: coverage >= 20 dBZ on '{args.basis}' below {args.min_cov20:.1%}"
          f"  ->  {pool['low_echo'].sum():,} / {n:,} low ({pool['low_echo'].mean():.1%})")
    print(f"kept for training: {pool['keep'].sum():,} / {n:,} ({pool['keep'].mean():.1%})"
          + (f", incl. {rescued.sum():,} low-echo kept at weight {1 / args.keep_low_frac:g}"
             if args.keep_low_frac else ""))
    print(f"samples whose target holds a value > {args.max_valid_dbz:g} dBZ: {pool['invalid_dbz'].sum():,}"
          + ("  (dropped)" if args.drop_invalid else "  (kept; --drop-invalid to drop)"))
    missing = (pool["n_target_frames"] < len(spec.offsets_min)).sum()
    print(f"samples with an incomplete target window (< {len(spec.offsets_min)} frames): {missing:,}")

    print(f"\nthreshold sensitivity ({args.basis} coverage >= 20 dBZ):")
    print(f"   {'min cov20':>9}  {'kept':>8}  {'kept %':>7}  " + "  ".join(f"{s:>7}" for s in SEASONS))
    for thr in SENSITIVITY:
        kept = basis >= thr
        per_season = [kept[pool["season"] == s].mean() if (pool["season"] == s).any() else np.nan
                      for s in SEASONS]
        mark = "  <-" if np.isclose(thr, args.min_cov20) else ""
        print(f"   {thr:>9.1%}  {kept.sum():>8,}  {kept.mean():>7.1%}  "
              + "  ".join(f"{p:>7.1%}" if np.isfinite(p) else f"{'-':>7}" for p in per_season) + mark)

    print("\nby season (pool -> kept):")
    for s in SEASONS:
        sel = pool["season"] == s
        if sel.any():
            print(f"   {s:<7} {sel.sum():>7,} -> {pool.loc[sel, 'keep'].sum():>7,}"
                  f"  ({pool.loc[sel, 'keep'].mean():.1%} kept, {sel.mean():.1%} -> "
                  f"{(sel & pool['keep']).sum() / max(pool['keep'].sum(), 1):.1%} of the set)")
    ty = pool["typhoon"] != ""
    if ty.any():
        print(f"   typhoon windows: {ty.sum():,} -> {pool.loc[ty, 'keep'].sum():,} kept")

    print("\nby month (pool -> kept):")
    for m, grp in pool.groupby("month"):
        print(f"   {m:>2}  {len(grp):>6,} -> {grp['keep'].sum():>6,}  ({grp['keep'].mean():.1%})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pool.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    if args.figure:
        plot_pool(pool, basis, args.min_cov20, args.figure)


def plot_pool(pool: pd.DataFrame, basis: pd.Series, min_cov20: float, out: str) -> None:
    """Neutral-titled two-panel figure: samples per month split into kept /
    dropped, and the kept share as the threshold varies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), layout="constrained", facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    months = range(1, 13)
    kept = pool[pool["keep"]].groupby("month").size().reindex(months, fill_value=0)
    dropped = pool[~pool["keep"]].groupby("month").size().reindex(months, fill_value=0)
    ax1.bar(months, kept, color=SPLIT_COLOR["train"], edgecolor=SURFACE, linewidth=2, label="kept")
    ax1.bar(months, dropped, bottom=kept, color=DROPPED_GREY, edgecolor=SURFACE, linewidth=2, label="dropped")
    _style(ax1)
    ax1.set_xticks(list(months))
    ax1.set_xlabel("month (local)", fontsize=12, color=INK_2)
    ax1.set_ylabel("samples", fontsize=12, color=INK_2)
    ax1.set_title(f"Samples per month (coverage >= 20 dBZ threshold {min_cov20:.0%})",
                  fontsize=13, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=11, loc="upper left")

    thr = np.linspace(0, 0.3, 61)
    share = [(basis >= t).mean() * 100 for t in thr]
    ax2.plot(thr * 100, share, color=SPLIT_COLOR["train"], linewidth=2)
    ax2.axvline(min_cov20 * 100, color=MUTED, linewidth=1.2, linestyle="--")
    _style(ax2, grid_axis="both")
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("threshold: coverage >= 20 dBZ (%)", fontsize=12, color=INK_2)
    ax2.set_ylabel("samples kept (%)", fontsize=12, color=INK_2)
    ax2.set_title("Samples kept vs. threshold", fontsize=13, color=INK, loc="left")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
