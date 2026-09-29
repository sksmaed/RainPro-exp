"""How much of the t0 radar field survives each downsampling operator, before
any model is involved.

`RainPro8Dataset` builds `radar_4km` / `radar_8km` by nearest-neighbor point
sampling of QPESUMS (~1.3 km native): ~1 of 9 native pixels per 4 km cell, ~1
of 36 per 8 km cell. For speckled convection that can drop most strong cores
before the network sees them. This script compares, on the target canvas
(`--size-km`, default 512 km around the configured centre):

    native (~1.3 km, reference) | 2 km NN (= target semantics)
    4 km NN (= current input) | 4 km Z-mean | 4 km max
    8 km NN (= current input) | 8 km Z-mean | 8 km max

NN uses the exact `NearestNeighborRegridder` the Dataset uses; Z-mean / max
use `rainpro.data.regrid.aggregate`. Every coarse grid's pixel centres coincide
with the model's own (larger) canvas of the same resolution, so "4 km NN"
here is literally the central part of `radar_4km`.

Per field it reports max dBZ, area >= t dBZ in km^2 (km^2, not pixel counts,
so resolutions are comparable), and the core hit rate: of the native pixels
>= 35 dBZ, the fraction whose containing coarse cell also reads >= 35 dBZ.
Per case, then pooled over cases (areas summed, hits pooled).

Decision rule this feeds: if 4 km NN loses much of the >= 35 dBZ area / hit
rate that 4 km max keeps, aliasing is real and the radar tiers should switch
to area aggregation before the next run.

Usage (CPU is fine, no checkpoint needed):
    python scripts/inspect_radar_downsampling.py \\
        --qpesums /work/kilin1203/datasets/QPESUMS/radar_data_combined.zarr \\
        --variable MaxDBZ --auto-top 8 --out figures/radar_downsampling.png \\
        --csv figures/radar_downsampling.csv
    # or explicit cases:
    python scripts/inspect_radar_downsampling.py ... --init-times 2021-06-04T06:00,2021-08-07T09:00
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainpro.data.rainpro8_dataset import (  # noqa: E402
    _extract_latlon,
    _fill_no_echo,
    _mask_missing,
    _open_store,
)
from rainpro.data.rainpro8_sources import (  # noqa: E402
    QPESUMS_MISSING_VALUES,
    QPESUMS_NO_ECHO_VALUES,
)
from rainpro.data.regrid import (  # noqa: E402
    KM_PER_DEG_LAT,
    NearestNeighborRegridder,
    aggregate,
    prepare_aggregation,
    target_grid,
)

THRESHOLDS = (20.0, 30.0, 35.0, 40.0)
CORE_DBZ = 35.0
PLOT_BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
FIELDS = [  # (label, resolution_km, operator); operator None == native
    ("native", None, None),
    ("2km NN", 2, "nn"),
    ("4km NN", 4, "nn"),
    ("4km Z-mean", 4, "zmean"),
    ("4km max", 4, "max"),
    ("8km NN", 8, "nn"),
    ("8km Z-mean", 8, "zmean"),
    ("8km max", 8, "max"),
]


def clean(frame: np.ndarray) -> np.ndarray:
    """Same sentinel handling as RainPro8Dataset: -99 -> 0 dBZ, -999 -> NaN."""
    frame = _fill_no_echo(frame, QPESUMS_NO_ECHO_VALUES, 0.0)
    return _mask_missing(frame, QPESUMS_MISSING_VALUES)


def pick_cases(ds, var, native_keep, n, start, end, step_hours, min_gap_hours):
    """Top-n times by native >= CORE_DBZ area inside the canvas, at least
    `min_gap_hours` apart so one long event doesn't fill every slot."""
    index = ds.indexes["time"]
    wanted = pd.date_range(start, end, freq=f"{step_hours}h")
    pos = index.get_indexer(wanted, method="nearest", tolerance=pd.Timedelta("5min"))
    pos = np.unique(pos[pos >= 0])
    scores = []
    for i in range(0, len(pos), 64):
        block = np.asarray(ds[var].isel(time=pos[i:i + 64].tolist()).values, dtype=np.float32)
        for p, frame in zip(pos[i:i + 64], block):
            vals = clean(frame).ravel()[native_keep]
            scores.append((int(np.sum(vals >= CORE_DBZ)), index[p]))
        print(f"  scanned {min(i + 64, len(pos))}/{len(pos)} frames", flush=True)
    chosen: list[pd.Timestamp] = []
    for score, t in sorted(scores, key=lambda s: -s[0]):
        if score == 0 or len(chosen) == n:
            break
        if all(abs(t - c) >= pd.Timedelta(hours=min_gap_hours) for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qpesums", required=True, help="QPESUMS zarr store (data_root['qpesums'])")
    ap.add_argument("--variable", default="MaxDBZ", help="raw variable name in the store")
    ap.add_argument("--init-times", default=None, help="comma-separated t0s; else --auto-top")
    ap.add_argument("--auto-top", type=int, default=8,
                     help="pick this many t0s with the largest native >=35 dBZ area in the canvas")
    ap.add_argument("--scan-start", default="2021-05-01")
    ap.add_argument("--scan-end", default="2021-10-01")
    ap.add_argument("--scan-step-hours", type=int, default=3)
    ap.add_argument("--min-gap-hours", type=int, default=24)
    ap.add_argument("--center-lat", type=float, default=23.7)
    ap.add_argument("--center-lon", type=float, default=121.0)
    ap.add_argument("--size-km", type=float, default=512.0, help="compared canvas (target_2km's)")
    ap.add_argument("--out", default="radar_downsampling.png")
    ap.add_argument("--csv", default=None, help="optional long-format CSV of every number printed")
    ap.add_argument("--max-plot-rows", type=int, default=6)
    args = ap.parse_args()

    ds = _open_store(args.qpesums, consolidated=True)
    src_lat, src_lon = _extract_latlon(ds)
    nn = NearestNeighborRegridder(src_lat, src_lon, ref_lat=args.center_lat)

    # Native reference: every native pixel inside the canvas, with its own area.
    native_map = prepare_aggregation(src_lat, src_lon, args.center_lat, args.center_lon, args.size_km, 1.0)
    native_keep = native_map.keep
    d_lat = np.abs(np.gradient(src_lat, axis=0)) * KM_PER_DEG_LAT
    d_lon = np.abs(np.gradient(src_lon, axis=1)) * KM_PER_DEG_LAT * np.cos(np.deg2rad(src_lat))
    native_area = (d_lat * d_lon).ravel()[native_keep]
    print(f"native pixels in the {args.size_km:g} km canvas: {native_keep.sum()} "
          f"(~{np.median(np.sqrt(native_area)):.2f} km)")

    grids = {}
    for _, res, op in FIELDS:
        if res is None or res in grids:
            continue
        dst_lat, dst_lon = target_grid(args.center_lat, args.center_lon, args.size_km, res)
        agg_map = prepare_aggregation(src_lat, src_lon, args.center_lat, args.center_lon, args.size_km, res)
        # Core hit rate indexes coarse cells by native-pixel order, which is
        # only valid if every resolution keeps the same native pixels (same
        # canvas bounds) -- true whenever size_km divides evenly by res.
        assert np.array_equal(agg_map.keep, native_keep), f"canvas bounds differ at {res} km"
        grids[res] = (nn.prepare(dst_lat, dst_lon), agg_map)

    if args.init_times:
        cases = [pd.Timestamp(t) for t in args.init_times.split(",")]
    else:
        print(f"scanning {args.scan_start}..{args.scan_end} every {args.scan_step_hours} h for cases ...")
        cases = pick_cases(ds, args.variable, native_keep, args.auto_top, args.scan_start,
                           args.scan_end, args.scan_step_hours, args.min_gap_hours)
    if not cases:
        raise SystemExit("no cases with any >= 35 dBZ echo in the canvas")

    index = ds.indexes["time"]
    rows: list[dict] = []  # long format, also the CSV
    pooled = {label: {"area": np.zeros(len(THRESHOLDS)), "hit": 0, "cores": 0, "max": -np.inf}
              for label, _, _ in FIELDS}
    plots = []

    for t in cases:
        p = index.get_indexer([t], method="nearest", tolerance=pd.Timedelta("5min"))[0]
        if p < 0:
            print(f"!! {t}: no QPESUMS frame within 5 min, skipped")
            continue
        frame = clean(np.asarray(ds[args.variable].isel(time=int(p)).values, dtype=np.float32))
        native_vals = frame.ravel()[native_keep]
        core = native_vals >= CORE_DBZ

        fields = {}
        for label, res, op in FIELDS:
            if res is None:
                continue
            nn_map, agg_map = grids[res]
            fields[label] = (nn.apply(frame, nn_map) if op == "nn" else aggregate(frame, agg_map, op), agg_map)

        print(f"\n=== t0 {t} ===")
        print(f"  {'field':<11} {'max':>5}  " + "  ".join(f"A>={th:g} km2".rjust(12) for th in THRESHOLDS)
              + f"  {'core hit':>9}")
        for label, res, _ in FIELDS:
            if res is None:
                vmax = float(np.nanmax(native_vals))
                areas = [float(native_area[native_vals >= th].sum()) for th in THRESHOLDS]
                hit = None
            else:
                field, agg_map = fields[label]
                vmax = float(np.nanmax(field))
                areas = [float(np.sum(field >= th) * res * res) for th in THRESHOLDS]
                # Coarse value of the cell containing each native core pixel.
                cell_vals = field.ravel()[agg_map.cell]
                hits = int(np.sum(cell_vals[core] >= CORE_DBZ))
                hit = hits / core.sum() if core.sum() else float("nan")
                pooled[label]["hit"] += hits
                pooled[label]["cores"] += int(core.sum())
            pooled[label]["area"] += areas
            pooled[label]["max"] = max(pooled[label]["max"], vmax)
            print(f"  {label:<11} {vmax:5.1f}  " + "  ".join(f"{a:12.0f}" for a in areas)
                  + (f"  {hit:9.2f}" if hit is not None else f"  {'(ref)':>9}"))
            for th, a in zip(THRESHOLDS, areas):
                rows.append({"t0": str(t), "field": label, "metric": f"area_ge{th:g}_km2", "value": a})
            rows.append({"t0": str(t), "field": label, "metric": "max_dbz", "value": vmax})
            if hit is not None:
                rows.append({"t0": str(t), "field": label, "metric": f"core_hit_ge{CORE_DBZ:g}", "value": hit})

        # Native crop for display: the canvas' pixels form a rectangle on the
        # regular QPESUMS grid.
        keep2d = native_keep.reshape(src_lat.shape)
        rr, cc = np.flatnonzero(keep2d.any(axis=1)), np.flatnonzero(keep2d.any(axis=0))
        native_img = frame[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        if src_lat[rr.min(), cc.min()] > src_lat[rr.max(), cc.min()]:
            native_img = native_img[::-1]  # rows north->south in the store; plot south-up
        plots.append((t, [native_img] + [fields[label][0] for label, res, _ in FIELDS if res is not None]))

    print(f"\n=== pooled over {len(plots)} cases ===")
    print(f"  {'field':<11} {'max':>5}  " + "  ".join(f"A>={th:g} km2".rjust(12) for th in THRESHOLDS)
          + f"  {'core hit':>9}  {'A>=35 / native':>14}")
    ref35 = pooled["native"]["area"][THRESHOLDS.index(CORE_DBZ)]
    for label, res, _ in FIELDS:
        pl = pooled[label]
        hit = pl["hit"] / pl["cores"] if pl["cores"] else float("nan")
        ratio = pl["area"][THRESHOLDS.index(CORE_DBZ)] / ref35 if ref35 else float("nan")
        print(f"  {label:<11} {pl['max']:5.1f}  " + "  ".join(f"{a:12.0f}" for a in pl["area"])
              + (f"  {hit:9.2f}" if res is not None else f"  {'(ref)':>9}") + f"  {ratio:14.2f}")
        rows.append({"t0": "pooled", "field": label, "metric": "area_ge35_ratio_to_native", "value": ratio})
        if res is not None:
            rows.append({"t0": "pooled", "field": label, "metric": f"core_hit_ge{CORE_DBZ:g}", "value": hit})

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t0", "field", "metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")

    plots = plots[: args.max_plot_rows]
    cmap = mpl.colormaps["turbo"].resampled(len(PLOT_BOUNDS) - 1)
    cmap.set_bad("0.85")
    norm = mpl.colors.BoundaryNorm(PLOT_BOUNDS, cmap.N)
    fig, axes = plt.subplots(len(plots), len(FIELDS), figsize=(2.6 * len(FIELDS), 2.6 * len(plots) + 0.8),
                             squeeze=False, layout="constrained")
    for r, (t, imgs) in enumerate(plots):
        for c, ((label, _, _), img) in enumerate(zip(FIELDS, imgs)):
            ax = axes[r][c]
            ax.imshow(np.ma.masked_invalid(img), cmap=cmap, norm=norm, origin="lower",
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=11)
            if c == 0:
                ax.set_ylabel(pd.Timestamp(t).strftime("%Y-%m-%d %H:%M"), fontsize=10)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                 orientation="horizontal", fraction=0.03, aspect=50, label="reflectivity (dBZ)",
                 ticks=PLOT_BOUNDS)
    fig.suptitle(f"t0 radar under each downsampling operator  |  {args.size_km:g} km canvas", fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
