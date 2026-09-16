"""What time ranges do the stores actually cover, and when is there real echo?

Two questions this answers before you try to run inference on a given day:

  1. Coverage. `data_root["qpesums"]` and `data_root["sta_h8"]` are independent
     archives with independent extents -- the converted STA_H8 zarr stores from
     `compress_sta_h8_taiwan.py` only cover whatever period you ran that script
     for, while the raw `.btp` archive may go much wider. An init_time is only
     usable if BOTH cover it (plus the source offsets around it, see
     `rainpro8_sources.build_taiwan_sources`).

  2. Echo. A forecast plotted over a clear-sky day shows nothing. This ranks
     candidate init_times by how much of the Taiwan domain is actually above a
     reflectivity threshold, so you can pick one worth visualising.

Only QPESUMS is read for the echo ranking (it's the target variable, and it's
~1 MB/frame against STA_H8's ~5.7 MB), at the store's native grid with no
regridding -- this is a scouting tool, not the training path.

Usage:
    python scripts/find_echo_times.py \\
        --data-root '{"qpesums": "...", "sta_h8": "..."}' \\
        --variable-aliases '{"max_dbz": "MaxDBZ"}' \\
        --date 2023-06-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainpro.data import sta_h8_raw
from rainpro.data.rainpro8_dataset import _is_zarr_store, split_sta_h8_paths
from rainpro.data.rainpro8_sources import QPESUMS_MISSING_VALUES, build_taiwan_sources


def report_coverage(data_root: dict[str, str]) -> tuple[pd.DatetimeIndex | None, pd.DatetimeIndex | None]:
    qpesums_times = sta_times = None

    path = data_root.get("qpesums")
    if path:
        with xr.open_zarr(path, consolidated=True) as ds:
            qpesums_times = pd.DatetimeIndex(ds["time"].values).sort_values()
        deltas = qpesums_times[1:] - qpesums_times[:-1]
        modal = pd.Series(deltas).mode()
        print(f"QPESUMS  {qpesums_times[0]} .. {qpesums_times[-1]}  "
              f"({len(qpesums_times)} steps, modal cadence {modal.iloc[0] if len(modal) else 'n/a'})")

    path = data_root.get("sta_h8")
    if path:
        parts = []
        for p in split_sta_h8_paths(path):
            if _is_zarr_store(p):
                with xr.open_zarr(p, consolidated=False) as ds:
                    parts.append(pd.DatetimeIndex(ds["time"].values))
                kind = "zarr"
            else:
                _, times = sta_h8_raw.scan_files(p)
                parts.append(pd.DatetimeIndex(times))
                kind = "raw .btp"
            print(f"STA_H8   {parts[-1].min()} .. {parts[-1].max()}  "
                  f"({len(parts[-1])} steps, {kind})  {os.path.basename(p.rstrip('/'))}")
        sta_times = parts[0]
        for part in parts[1:]:
            sta_times = sta_times.union(part)
        sta_times = sta_times.sort_values()

    return qpesums_times, sta_times


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="same JSON dict as --data.data_root")
    ap.add_argument("--variable-aliases", default="{}")
    ap.add_argument("--date", default=None,
                     help="day to scan for echo, YYYY-MM-DD (omit to only report coverage)")
    ap.add_argument("--scan-every-min", type=int, default=30, help="candidate init_time spacing")
    ap.add_argument("--threshold-dbz", type=float, default=20.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--center-lat", type=float, default=23.7)
    ap.add_argument("--center-lon", type=float, default=121.0)
    ap.add_argument("--radius-deg", type=float, default=1.5,
                     help="echo fraction is measured within this box around the center, so a "
                          "storm far offshore doesn't score as a good visualisation target")
    args = ap.parse_args()

    data_root = json.loads(args.data_root)
    aliases = json.loads(args.variable_aliases)

    print("=== store coverage ===")
    qpesums_times, sta_times = report_coverage(data_root)
    if qpesums_times is not None and sta_times is not None:
        lo, hi = max(qpesums_times[0], sta_times[0]), min(qpesums_times[-1], sta_times[-1])
        print(f"\noverlap  {lo} .. {hi}" if lo <= hi else "\n!! NO OVERLAP between the two archives")

    if args.date is None:
        return

    day = pd.Timestamp(args.date)
    sat_offsets = build_taiwan_sources(include_satellite=True)["satellite_8km"].offsets_min
    print(f"\n=== echo scan for {day.date()} (>= {args.threshold_dbz} dBZ "
          f"within +-{args.radius_deg} deg of {args.center_lat},{args.center_lon}) ===")

    var = aliases.get("max_dbz", "max_dbz")
    with xr.open_zarr(data_root["qpesums"], consolidated=True) as ds:
        if var not in ds:
            raise SystemExit(f"'{var}' not in the QPESUMS store; has {list(ds.data_vars)}. "
                             f"Pass --variable-aliases to map it.")
        index = pd.DatetimeIndex(ds["time"].values)
        lat_name = "lat" if "lat" in ds.coords or "lat" in ds else "latitude"
        lon_name = "lon" if "lon" in ds.coords or "lon" in ds else "longitude"
        lat, lon = np.asarray(ds[lat_name].values), np.asarray(ds[lon_name].values)
        # native grid is regular 1-D lat/lon -- take a box, no regridding needed
        rows = np.where(np.abs(lat - args.center_lat) <= args.radius_deg)[0]
        cols = np.where(np.abs(lon - args.center_lon) <= args.radius_deg)[0]
        box = dict(zip(ds[var].dims[1:], (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))))

        candidates = pd.date_range(
            day, day + pd.Timedelta(days=1), freq=f"{args.scan_every_min}min", inclusive="left"
        )
        pos = index.get_indexer(candidates, method="nearest", tolerance=pd.Timedelta("5min"))
        usable = pos >= 0
        if not usable.any():
            raise SystemExit(f"QPESUMS has no timestep within 5 min of any candidate on {day.date()}")

        block = np.asarray(ds[var].isel(time=pos[usable]).isel(**box).values, dtype=np.float32)

    for sentinel in QPESUMS_MISSING_VALUES:
        block = np.where(np.isclose(block, sentinel), np.nan, block)
    frac = np.nanmean(block >= args.threshold_dbz, axis=(1, 2))
    peak = np.nanmax(block, axis=(1, 2))

    times = candidates[usable]
    order = np.argsort(-frac)
    print(f"{'init_time':<22}{'echo frac':>11}{'peak dBZ':>11}   satellite coverage")
    for i in order[: args.top]:
        t = times[i]
        if sta_times is not None:
            need = [t + pd.Timedelta(minutes=o) for o in sat_offsets]
            got = sta_times.get_indexer(pd.DatetimeIndex(need), method="nearest",
                                        tolerance=pd.Timedelta("31min"))
            sat = "OK" if (got >= 0).all() else f"MISSING {[str(n) for n, g in zip(need, got) if g < 0]}"
        else:
            sat = "(not checked)"
        print(f"{str(t):<22}{frac[i]:>10.1%}{peak[i]:>11.1f}   {sat}")


if __name__ == "__main__":
    main()
