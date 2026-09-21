"""Inspect a zarr store's time range, temporal/spatial resolution, and -- for a
given time window -- how complete the data actually is.

Usage:
    python scripts/inspect_zarr.py <path-to-zarr> [<path-to-zarr> ...] [options]

Examples:
    # whole-store overview (as before)
    python scripts/inspect_zarr.py ~/work/kilin1203/datasets/QPESUMS/radar_data_combined.zarr

    # 2025 年 7 月的資料狀態：缺哪些時刻、每天有幾張
    python scripts/inspect_zarr.py <zarr> --period 2025-07

    # 同上，外加逐張影像的哨兵值統計（-999 未觀測 / -99 無回波）
    python scripts/inspect_zarr.py <zarr> --period 2025-07 --stats

    # 任意區間
    python scripts/inspect_zarr.py <zarr> --start 2025-07-15 --end 2025-07-20
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

LATLON_CANDIDATES = [("lat", "lon"), ("XLAT", "XLONG"), ("latitude", "longitude")]

# QPESUMS sentinels (see rainpro/data/rainpro8_sources.py and
# docs/rainpro_dataset.md): -999 = never observed, -99 = observed but no echo.
DEFAULT_MISSING_VALUES = (-999.0,)
DEFAULT_NO_ECHO_VALUES = (-99.0,)


def _find_latlon(ds: xr.Dataset) -> tuple[str, str] | None:
    for lat_name, lon_name in LATLON_CANDIDATES:
        if lat_name in ds.variables and lon_name in ds.variables:
            return lat_name, lon_name
    return None


def _describe_resolution(values: np.ndarray, unit: str) -> str:
    diffs = np.diff(np.sort(np.unique(values)))
    if diffs.size == 0:
        return "n/a (single value)"
    uniq_diffs = np.unique(np.round(diffs, 6))
    if uniq_diffs.size == 1:
        return f"{uniq_diffs[0]:g} {unit} (uniform)"
    return (
        f"non-uniform: min={diffs.min():g}, max={diffs.max():g}, "
        f"median={np.median(diffs):g} {unit}"
    )


def _parse_bound(text: str, *, side: str) -> pd.Timestamp:
    """Parse a (possibly partial) timestamp into an inclusive window bound.

    Partial specs expand to cover the whole period they name, so
    ``--end 2025-07`` means "end of July", not "1 July 00:00".
    """
    try:
        period = pd.Period(text)
    except Exception:
        ts = pd.Timestamp(text)
        return ts
    return period.start_time if side == "start" else period.end_time


def _resolve_window(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if args.period:
        period = pd.Period(args.period)
        return period.start_time, period.end_time
    if args.start or args.end:
        start = _parse_bound(args.start, side="start") if args.start else pd.Timestamp.min
        end = _parse_bound(args.end, side="end") if args.end else pd.Timestamp.max
        return start, end
    return None


def _modal_delta(time: pd.DatetimeIndex) -> pd.Timedelta | None:
    if len(time) < 2:
        return None
    deltas = pd.Series(time[1:] - time[:-1])
    return deltas.mode().iloc[0]


def _runs(missing: pd.DatetimeIndex, step: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Collapse a sorted list of missing timestamps into contiguous runs."""
    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for ts in missing:
        if runs and ts - runs[-1][1] == step:
            start, _, n = runs[-1]
            runs[-1] = (start, ts, n + 1)
        else:
            runs.append((ts, ts, 1))
    return runs


def _sentinel_mask(da: xr.DataArray, values: tuple[float, ...]) -> xr.DataArray:
    mask = xr.zeros_like(da, dtype=bool)
    for value in values:
        mask = mask | (da == value)
    return mask


def _frame_stats(
    ds: xr.Dataset,
    var: str,
    missing_values: tuple[float, ...],
    no_echo_values: tuple[float, ...],
) -> pd.DataFrame:
    """Per-timestep sentinel fractions and valid-value stats (one dask pass)."""
    da = ds[var]
    space_dims = [d for d in da.dims if d != "time"]

    missing = _sentinel_mask(da, missing_values) | da.isnull()
    no_echo = _sentinel_mask(da, no_echo_values)
    valid = da.where(~missing & ~no_echo)

    out = xr.Dataset(
        {
            "missing_frac": missing.mean(dim=space_dims),
            "no_echo_frac": no_echo.mean(dim=space_dims),
            "echo_frac": (~missing & ~no_echo).mean(dim=space_dims),
            "valid_max": valid.max(dim=space_dims),
            "valid_mean": valid.mean(dim=space_dims),
        }
    ).compute()

    frame = out.to_dataframe()
    return frame.set_index(pd.DatetimeIndex(out["time"].values))


def inspect_window(
    ds: xr.Dataset,
    window: tuple[pd.Timestamp, pd.Timestamp],
    args: argparse.Namespace,
) -> None:
    start, end = window
    print(f"\n{'-' * 80}")
    print(f"-- window: {start} .. {end} --")

    if "time" not in ds.coords and "time" not in ds.variables:
        print("  no 'time' coordinate -- cannot slice by time")
        return

    full_time = pd.DatetimeIndex(ds["time"].values)
    sub = ds.sel(time=slice(start, end))
    time = pd.DatetimeIndex(sub["time"].values)

    if len(time) == 0:
        print("  NO data in this window")
        print(f"  store covers {full_time.min()} .. {full_time.max()}")
        return

    # Does the store even reach across the requested window?
    if full_time.min() > start:
        print(f"  ! store starts at {full_time.min()}, after window start")
    if full_time.max() < end:
        print(f"  ! store ends at {full_time.max()}, before window end")

    step = pd.Timedelta(args.freq) if args.freq else _modal_delta(time) or _modal_delta(full_time)
    print(f"  cadence used: {step}" + ("" if args.freq else " (modal, inferred)"))
    print(f"  first: {time.min()}")
    print(f"  last:  {time.max()}")

    duplicates = time[time.duplicated()]
    if len(duplicates):
        print(f"  ! {len(duplicates)} duplicate timestamp(s), e.g. {duplicates[:5].tolist()}")

    missing = pd.DatetimeIndex([])
    if step is not None and step > pd.Timedelta(0):
        grid_start = time.min().ceil(step)
        expected = pd.date_range(grid_start, min(end, time.max()), freq=step)
        missing = expected.difference(time)
        present = len(expected) - len(missing)
        pct = 100.0 * present / len(expected) if len(expected) else float("nan")
        print(f"  expected steps: {len(expected)}  present: {present}  missing: {len(missing)}")
        print(f"  completeness: {pct:.2f}%")
        off_grid = time.difference(expected)
        if len(off_grid):
            print(
                f"  ! {len(off_grid)} timestamp(s) off the {step} grid, "
                f"e.g. {off_grid[:5].tolist()}"
            )
    else:
        print(f"  steps in window: {len(time)} (cadence unknown, cannot check for gaps)")

    if len(missing):
        runs = _runs(missing, step)
        print(f"\n  -- missing runs ({len(runs)}) --")
        for run_start, run_end, n in runs[: args.max_list]:
            if n == 1:
                print(f"    {run_start}                        (1 step)")
            else:
                print(f"    {run_start} .. {run_end}  ({n} steps, {n * step})")
        if len(runs) > args.max_list:
            print(f"    ... and {len(runs) - args.max_list} more")

    stats: pd.DataFrame | None = None
    if args.stats:
        var = args.var or (next(iter(ds.data_vars)) if ds.data_vars else None)
        if var is None:
            print("\n  -- value stats --\n    no data variables")
        else:
            print(f"\n  -- value stats for '{var}' (this loads the window, may take a while) --")
            stats = _frame_stats(sub, var, tuple(args.missing_values), tuple(args.no_echo_values))
            print(f"    missing ({', '.join(map(str, args.missing_values))} or NaN): "
                  f"mean {100 * stats['missing_frac'].mean():.3f}% of pixels/frame")
            print(f"    no-echo ({', '.join(map(str, args.no_echo_values))}): "
                  f"mean {100 * stats['no_echo_frac'].mean():.3f}% of pixels/frame")
            print(f"    echo (valid, non-sentinel): "
                  f"mean {100 * stats['echo_frac'].mean():.3f}% of pixels/frame")
            print(f"    valid value range: {np.nanmin(stats['valid_max']):.2f} .. "
                  f"{np.nanmax(stats['valid_max']):.2f} (per-frame max)")

            blank = stats.index[stats["echo_frac"] == 0]
            print(f"    frames with zero echo pixels: {len(blank)}")
            for ts in blank[: args.max_list]:
                print(f"      {ts}")
            if len(blank) > args.max_list:
                print(f"      ... and {len(blank) - args.max_list} more")

            all_missing = stats.index[stats["missing_frac"] == 1.0]
            if len(all_missing):
                print(f"    ! frames that are entirely missing: {len(all_missing)}")
                for ts in all_missing[: args.max_list]:
                    print(f"      {ts}")

    if not args.no_daily:
        print("\n  -- per-day inventory --")
        counts = pd.Series(1, index=time).resample("1D").sum()
        per_day = int(round(pd.Timedelta("1D") / step)) if step else None
        header = "    date        frames"
        if per_day:
            header += f" / {per_day:<5d} missing"
        if stats is not None:
            header += "   echo%   miss%   maxval"
        print(header)
        daily_stats = stats.resample("1D").mean() if stats is not None else None
        daily_max = stats["valid_max"].resample("1D").max() if stats is not None else None
        for day, n in counts.items():
            line = f"    {day.date()}  {int(n):6d}"
            if per_day:
                line += f" / {per_day:<5d} {per_day - int(n):7d}"
            if daily_stats is not None:
                row = daily_stats.loc[day]
                maxval = daily_max.loc[day]
                line += (
                    f"  {100 * row['echo_frac']:6.3f}  {100 * row['missing_frac']:6.3f}"
                    f"  {maxval:7.1f}"
                )
            print(line)


def inspect(path: str, args: argparse.Namespace) -> None:
    print(f"\n{'=' * 80}\n{path}\n{'=' * 80}")
    try:
        ds = xr.open_zarr(path, consolidated=True)
    except Exception:
        ds = xr.open_zarr(path, consolidated=False)

    print("\n-- dims --")
    for dim, size in ds.sizes.items():
        print(f"  {dim}: {size}")

    print("\n-- data variables --")
    for name, da in ds.data_vars.items():
        print(f"  {name}: dims={da.dims}, shape={da.shape}, dtype={da.dtype}")

    if "time" in ds.coords or "time" in ds.variables:
        time = pd.DatetimeIndex(ds["time"].values)
        print("\n-- time range --")
        print(f"  start: {time.min()}")
        print(f"  end:   {time.max()}")
        print(f"  count: {len(time)}")
        if len(time) > 1:
            deltas = time[1:] - time[:-1]
            unique_deltas = pd.unique(deltas)
            if len(unique_deltas) == 1:
                print(f"  cadence: {unique_deltas[0]} (uniform)")
            else:
                print(
                    f"  cadence: non-uniform, min={deltas.min()}, "
                    f"max={deltas.max()}, median={deltas.median()}"
                )
                gaps = deltas[deltas != deltas.min()]
                print(f"  {len(gaps)} irregular gap(s), e.g. {gaps[:5].tolist()}")
    else:
        print("\n-- time range --\n  no 'time' coordinate found")

    latlon = _find_latlon(ds)
    if latlon:
        lat_name, lon_name = latlon
        lat = np.asarray(ds[lat_name].values)
        lon = np.asarray(ds[lon_name].values)
        print(f"\n-- spatial extent ({lat_name}/{lon_name}) --")
        print(f"  lat range: [{lat.min():.4f}, {lat.max():.4f}]")
        print(f"  lon range: [{lon.min():.4f}, {lon.max():.4f}]")
        print(f"  lat resolution: {_describe_resolution(lat, 'deg')}")
        print(f"  lon resolution: {_describe_resolution(lon, 'deg')}")
    else:
        print("\n-- spatial extent --\n  no recognized lat/lon coordinate found")
        print(f"  coords available: {list(ds.coords)}")

    print("\n-- chunking (first data var) --")
    if ds.data_vars:
        first_var = next(iter(ds.data_vars.values()))
        print(f"  {first_var.encoding.get('chunks', 'unknown')}")

    window = _resolve_window(args)
    if window is not None:
        inspect_window(ds, window, args)

    ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="Path(s) to zarr store(s)")
    parser.add_argument(
        "--period",
        help="Time window as a single period, e.g. 2025 / 2025-07 / 2025-07-15 "
        "(shorthand for --start/--end)",
    )
    parser.add_argument("--start", help="Window start (inclusive), e.g. 2025-07-01 or 2025-07-01T06:00")
    parser.add_argument(
        "--end",
        help="Window end (inclusive); partial specs expand, so --end 2025-07 means end of July",
    )
    parser.add_argument(
        "--freq",
        help="Expected cadence for gap detection, e.g. 10min (default: modal cadence in window)",
    )
    parser.add_argument(
        "--no-daily",
        action="store_true",
        help="Skip the per-day frame inventory (printed by default for a window)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute per-frame sentinel/value stats in the window (reads the data)",
    )
    parser.add_argument("--var", help="Variable used by --stats (default: first data variable)")
    parser.add_argument(
        "--missing-values",
        type=float,
        nargs="*",
        default=list(DEFAULT_MISSING_VALUES),
        help="Sentinel(s) meaning 'not observed' (default: -999)",
    )
    parser.add_argument(
        "--no-echo-values",
        type=float,
        nargs="*",
        default=list(DEFAULT_NO_ECHO_VALUES),
        help="Sentinel(s) meaning 'observed, no echo' (default: -99)",
    )
    parser.add_argument(
        "--max-list", type=int, default=20, help="Max rows to print per list (default 20)"
    )
    args = parser.parse_args()

    for path in args.paths:
        inspect(path, args)


if __name__ == "__main__":
    sys.exit(main())
