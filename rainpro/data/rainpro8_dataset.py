"""Torch Dataset for RainPro-8 (Taiwan), reading QPESUMS / STA_H8 (and
optionally GFS, see `include_gfs` in `rainpro8_sources.py`) from zarr stores
and assembling the per-tier (4km/8km/16km) input tensors and `target_2km`
expected by `rainpro.network.rainpro8.RainPro` (see `rainpro8.ipynb`).

Each source is read lazily via xarray/zarr and regridded with nearest-neighbor
lookup (`rainpro.data.regrid`) onto a common square canvas per tier, centered
on `center_lat`/`center_lon` with optional random spatial jitter for training
augmentation (paper Sec. A.3: +/-256 km).

Exact store layouts (variable names, static coordinate names) vary by
pipeline; `variable_aliases` lets the Taiwan zarr's real names be mapped onto
the source-spec's canonical variable names without touching
`rainpro8_sources.py`.
"""

from __future__ import annotations

import datetime
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from rainpro.data.marshall_palmer import dbz_to_mmh
from rainpro.data.normalize import DEFAULT_NORM_BOUNDS, minmax_normalize
from rainpro.data.regrid import NearestNeighborRegridder, target_grid
from rainpro.data.rainpro8_sources import SourceSpec

# Which zarr store (key into `data_root`) each source is read from.
SOURCE_STORE = {
    "target_2km": "qpesums",
    "radar_4km": "qpesums",
    "radar_8km": "qpesums",
    "satellite_8km": "sta_h8",
    "gfs_16km": "gfs",
    "gfs_forecast_16km": "gfs_forecast",
}

TIME_TOLERANCE = {
    "target_2km": "5min",
    "radar_4km": "5min",
    "radar_8km": "5min",
    "satellite_8km": "8min",
    "gfs_16km": "3h",
    "gfs_forecast_16km": "31min",
}

# Candidate (lat, lon) variable/coord names to try, in order, per store. QPESUMS
# uses a regular 0.0125deg lat/lon grid; STA_H8's lat/lon lookup table variable
# name is store-specific and should be added via `latlon_names` if it differs
# (see docs/rainpro_dataset.md: "STA_H8_Plt_IR_for_glbdisplay 有逐像素經緯度查找表").
DEFAULT_LATLON_CANDIDATES = [("lat", "lon"), ("XLAT", "XLONG"), ("latitude", "longitude")]


class RainPro8Dataset(Dataset):
    def __init__(
        self,
        data_root: dict[str, str],
        sources: dict[str, SourceSpec],
        init_times: Sequence[np.datetime64 | datetime.datetime],
        center_lat: float = 23.7,
        center_lon: float = 121.0,
        jitter_km: float = 0.0,
        norm_bounds: dict[str, tuple[float, float]] | None = None,
        variable_aliases: dict[str, str] | None = None,
        latlon_names: dict[str, tuple[str, str]] | None = None,
        fill_value: float = 0.0,
        rng_seed: int = 0,
    ):
        self.data_root = data_root
        self.sources = sources
        self.init_times = list(init_times)
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.jitter_km = jitter_km
        self.norm_bounds = {**DEFAULT_NORM_BOUNDS, **(norm_bounds or {})}
        self.variable_aliases = variable_aliases or {}
        self.latlon_names = latlon_names or {}
        self.fill_value = fill_value
        self.rng_seed = rng_seed
        self.epoch = 0  # bump via `set_epoch()` (e.g. from a Lightning hook) to vary
        # augmentation across epochs; not required for correctness.

        self._datasets: dict[str, xr.Dataset] = {}
        self._regridders: dict[str, NearestNeighborRegridder] = {}

    def __len__(self) -> int:
        return len(self.init_times)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rng_for_index(self, index: int) -> np.random.Generator:
        # A per-(seed, epoch, index) RNG rather than a single `self._rng` consumed
        # sequentially: with `num_workers > 0`, each worker process gets its own
        # fork of this Dataset *after* `__init__` runs, so a shared, already-seeded
        # generator would be replayed identically (and hence duplicated) by every
        # worker instead of decorrelated across them. Keying by index also makes
        # augmentation deterministic per sample, independent of worker scheduling.
        return np.random.default_rng((self.rng_seed, self.epoch, index))

    def close(self):
        for ds in self._datasets.values():
            ds.close()
        self._datasets = {}

    # -- lazy zarr access -------------------------------------------------

    def _get_store(self, store_key: str) -> xr.Dataset:
        if store_key not in self._datasets:
            path = self.data_root.get(store_key)
            if path is None:
                raise KeyError(
                    f"data_root is missing a zarr path for '{store_key}'; "
                    f"got keys {list(self.data_root)}"
                )
            self._datasets[store_key] = xr.open_zarr(path, consolidated=True)
        return self._datasets[store_key]

    def _get_regridder(self, store_key: str) -> NearestNeighborRegridder:
        if store_key not in self._regridders:
            ds = self._get_store(store_key)
            candidates = (
                [self.latlon_names[store_key]] + DEFAULT_LATLON_CANDIDATES
                if store_key in self.latlon_names
                else DEFAULT_LATLON_CANDIDATES
            )
            lat, lon = _extract_latlon(ds, candidates)
            self._regridders[store_key] = NearestNeighborRegridder(
                lat, lon, ref_lat=self.center_lat
            )
        return self._regridders[store_key]

    # -- sample assembly ----------------------------------------------------

    def _sample_center(self, rng: np.random.Generator) -> tuple[float, float]:
        if self.jitter_km <= 0:
            return self.center_lat, self.center_lon
        dlat_km, dlon_km = rng.uniform(-self.jitter_km, self.jitter_km, size=2)
        km_per_deg_lon = 111.32 * np.cos(np.deg2rad(self.center_lat))
        return (
            self.center_lat + dlat_km / 111.32,
            self.center_lon + dlon_km / km_per_deg_lon,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        init_time = np.datetime64(self.init_times[index])
        rng = self._rng_for_index(index)
        center_lat, center_lon = self._sample_center(rng)

        sample: dict[str, torch.Tensor] = {}
        for name, spec in self.sources.items():
            store_key = SOURCE_STORE[name]
            ds = self._get_store(store_key)
            regridder = self._get_regridder(store_key)
            dst_lat, dst_lon = target_grid(center_lat, center_lon, spec.size_km, spec.resolution_km)

            is_target = name == "target_2km"
            frames = [
                self._read_frame(
                    ds, regridder, spec, init_time, offset, dst_lat, dst_lon, keep_nan=is_target
                )
                for offset in spec.offsets_min
            ]
            # (T, C, H, W)
            arr = np.stack(frames, axis=0)

            if is_target:
                arr = dbz_to_mmh(arr)  # keep NaN (missing) as NaN through the conversion;
                # OrdinalConsistentLoss masks pixels equal to the NaN sentinel (see
                # rainpro.loss.ordinal_consistent.Bucketize).

            sample[name] = torch.from_numpy(arr).float()

        return sample

    def _read_frame(
        self,
        ds: xr.Dataset,
        regridder: NearestNeighborRegridder,
        spec: SourceSpec,
        init_time: np.datetime64,
        offset_min: int,
        dst_lat: np.ndarray,
        dst_lon: np.ndarray,
        keep_nan: bool = False,
    ) -> np.ndarray:
        """Returns (C, H, W) for one timestep of one source."""
        all_vars = list(spec.variables) + list(spec.variables_3d)
        raw_names = [self.variable_aliases.get(v, v) for v in all_vars]
        # A source is "static" if none of its variables actually carry a time
        # dimension in the store, even if the store also holds other,
        # time-varying variables. Static sources skip time selection entirely.
        is_static = not any("time" in ds[n].dims for n in raw_names)

        if is_static:
            frame_ds = ds
        else:
            query_time = init_time + np.timedelta64(offset_min, "m")
            try:
                frame_ds = ds.sel(time=query_time, method="nearest", tolerance=_tolerance(spec))
            except KeyError:
                # No timestep within tolerance (e.g. sensor outage): treat as missing.
                out = np.full((spec.channels, *dst_lat.shape), np.nan, dtype=np.float32)
                return out if keep_nan else np.where(np.isnan(out), self.fill_value, out)

        channels = []
        for var in spec.variables:
            raw_name = self.variable_aliases.get(var, var)
            data = np.asarray(frame_ds[raw_name].values, dtype=np.float32)
            regridded = regridder(data, dst_lat, dst_lon, fill_value=np.nan)
            regridded = minmax_normalize(regridded, self.norm_bounds.get(var))
            channels.append(regridded)

        for var in spec.variables_3d:
            raw_name = self.variable_aliases.get(var, var)
            data = np.asarray(frame_ds[raw_name].values, dtype=np.float32)  # (level, y, x)
            data = data[list(spec.levels)]
            regridded = regridder(data, dst_lat, dst_lon, fill_value=np.nan)
            regridded = minmax_normalize(regridded, self.norm_bounds.get(var))
            channels.extend(regridded)

        out = np.stack(channels, axis=0)
        if not keep_nan:
            out = np.where(np.isnan(out), self.fill_value, out)
        return out


def _tolerance(spec: SourceSpec) -> pd.Timedelta:
    return pd.Timedelta(TIME_TOLERANCE.get(spec.name, "31min"))


def _extract_latlon(
    ds: xr.Dataset, candidates: Sequence[tuple[str, str]] = DEFAULT_LATLON_CANDIDATES
) -> tuple[np.ndarray, np.ndarray]:
    """Supports both 1D regular-grid coords (`lat(lat)`, `lon(lon)`) and 2D
    per-pixel lat/lon fields (e.g. STA_H8's lookup table)."""
    for lat_name, lon_name in candidates:
        if lat_name in ds.variables and lon_name in ds.variables:
            lat, lon = ds[lat_name], ds[lon_name]
            if lat.ndim == 1 and lon.ndim == 1:
                lat_grid, lon_grid = np.meshgrid(lat.values, lon.values, indexing="ij")
                return lat_grid, lon_grid
            # drop any leading (e.g. time) dims: coordinates are assumed static
            lat_vals = np.asarray(lat.values)
            lon_vals = np.asarray(lon.values)
            return lat_vals[(0,) * (lat_vals.ndim - 2)], lon_vals[(0,) * (lon_vals.ndim - 2)]

    raise KeyError(
        f"Could not find lat/lon coordinates among {[v for v in ds.variables]}; "
        "pass `latlon_names` to RainPro8Dataset to specify them explicitly."
    )
