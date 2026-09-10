"""Lazy reader for raw STA_H8 `.btp` files, standing in for a zarr store.

STA_H8 (Himawari-8/9 IR brightness temperature, 2750x2750 LCC, 9 bands
B08-B16, hourly -- see docs/rainpro_dataset.md and the real-archive findings
in scripts/inspect_sta_h8_times.py) has never been converted to zarr: the
source directory is read-only here, and even a compressed year would be
several hundred GB to a few TB (per the reference conversion project below),
far past a typical local/home quota. Rather than writing a converted copy
anywhere, this reads `.btp` files directly and on demand, wrapped in a lazy
dask-backed `xr.Dataset` so `RainPro8Dataset._read_frame`'s existing
`ds.sel(time=..., method="nearest", tolerance=...)` / `frame_ds[var].values`
code path works completely unmodified -- only `RainPro8Dataset._get_store`
needs to call `open_sta_h8_raw()` instead of `xr.open_zarr()` for this one
store. Each `.values` access triggers exactly one `.btp` file read (per
band, per timestep actually requested), not an eager load of the archive.

Raw format confirmed against real data by
github.com/sksmaed/weather_data_compression's compress/sta_h8.py and
compress/sta_h8_latlon.py (cross-checked against
docs/STA_H8_Plt_IR_for_glbdisplay.f90):
  - one band file = one unformatted direct-access record, IX*IY float32
    values, no header, i (x) fastest, j (y) counting DOWN from IY to 1 --
    i.e. a plain `np.fromfile(path, dtype="<f4").reshape(IY, IX)`, no row
    flip needed.
  - filename: `{YYYY-MM-DD}_{HHMM}.{Band}.LCC.btp`, e.g.
    `2021-06-01_0200.B08.LCC.btp`, under an arbitrary directory nesting
    (this module doesn't assume a depth, same as
    scripts/inspect_sta_h8_times.py).
  - static per-pixel lat/lon lookup table: a separate binary file, one
    record holding two IX*IY float32 blocks (lat then lon), same row order
    as the band files. Vendored at docs/STA_H8_Proj_Scale050_2750x2750
    (copied from the same reference repo, since it's small/static and this
    project doesn't have write access to generate its own copy).
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

IX = 2750
IY = 2750
BANDS = [f"B{n:02d}" for n in range(8, 17)]  # B08..B16

DEFAULT_LATLON_PATH = str(
    Path(__file__).resolve().parents[2] / "docs" / "STA_H8_Proj_Scale050_2750x2750"
)

_TS_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})_(\d{2})(\d{2})")  # YYYY-MM-DD_HHMM
_BAND_RE = re.compile(r"\.(B0[8-9]|B1[0-6])\.")


def parse_ts_band(path: str) -> tuple[datetime | None, str | None]:
    name = os.path.basename(path)
    m = _TS_RE.search(name)
    bm = _BAND_RE.search(name)
    if not m or not bm:
        return None, None
    y, mo, d, h, _mi = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h), bm.group(1)
    except ValueError:
        return None, None


def load_band_frame(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != IX * IY:
        raise ValueError(f"{path}: expected {IX * IY} float32 values, got {raw.size}")
    return raw.reshape(IY, IX)


def load_sta_h8_latlon(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (lat, lon), each (IY, IX) float32, same row order as `load_band_frame`."""
    raw = np.fromfile(path, dtype="<f4")
    expected = 2 * IX * IY
    if raw.size != expected:
        raise ValueError(f"{path}: expected {expected} float32 values, got {raw.size}")
    lat = raw[: IX * IY].reshape(IY, IX)
    lon = raw[IX * IY :].reshape(IY, IX)
    return lat, lon


def scan_files(root: str) -> tuple[dict[tuple[pd.Timestamp, str], str], list[pd.Timestamp]]:
    """Walks `root` for `.btp` files. Returns ({(time, band): path}, sorted unique times)."""
    file_index: dict[tuple[pd.Timestamp, str], str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".btp"):
                continue
            ts, band = parse_ts_band(fname)
            if ts is None:
                continue
            file_index[(pd.Timestamp(ts), band)] = os.path.join(dirpath, fname)
    times = sorted({t for t, _ in file_index})
    return file_index, times


def _load_or_nan(path: str | None) -> np.ndarray:
    if path is None:
        return np.full((IY, IX), np.nan, dtype=np.float32)
    return load_band_frame(path)


_delayed_load = dask.delayed(_load_or_nan)


def open_sta_h8_raw(root: str, latlon_path: str = DEFAULT_LATLON_PATH) -> xr.Dataset:
    """Lazy, dask-backed `xr.Dataset`: one (time, y, x) variable per band
    (B08..B16) plus static (y, x) `lat`/`lon` coords. A missing (time, band)
    file comes back as NaN, same convention as `RainPro8Dataset._mask_missing`
    /QPESUMS's missing-value handling downstream, rather than being skipped."""
    file_index, times = scan_files(root)
    if not times:
        raise FileNotFoundError(f"No STA_H8 .btp files found under {root!r}")
    lat, lon = load_sta_h8_latlon(latlon_path)

    data_vars = {}
    for band in BANDS:
        blocks = [
            da.from_delayed(
                _delayed_load(file_index.get((t, band))), shape=(IY, IX), dtype=np.float32
            )
            for t in times
        ]
        data_vars[band] = (("time", "y", "x"), da.stack(blocks, axis=0))

    return xr.Dataset(
        data_vars,
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
        },
    )
