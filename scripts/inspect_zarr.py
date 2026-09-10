"""Inspect a zarr store's time range, temporal resolution, and spatial resolution.

Usage:
    python scripts/inspect_zarr.py <path-to-zarr> [<path-to-zarr> ...]

Example:
    python scripts/inspect_zarr.py ~/work/kilin1203/datasets/QPESUMS/radar_data_combined.zarr
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

LATLON_CANDIDATES = [("lat", "lon"), ("XLAT", "XLONG"), ("latitude", "longitude")]


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


def inspect(path: str) -> None:
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

    ds.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Path(s) to zarr store(s)")
    args = parser.parse_args()

    for path in args.paths:
        inspect(path)


if __name__ == "__main__":
    sys.exit(main())
