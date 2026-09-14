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
import os
from collections import OrderedDict
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from rainpro.data import sta_h8_raw
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
    # STA_H8's real archive is hourly, not the originally-assumed 10-minute
    # cadence (see `build_taiwan_sources`'s `satellite_8km` docstring note) --
    # widened to match `gfs_forecast_16km` below, the other hourly source, so a
    # query offset that doesn't land exactly on the hour still snaps to the
    # nearest real timestamp instead of always missing by design.
    "satellite_8km": "31min",
    "gfs_16km": "3h",
    "gfs_forecast_16km": "31min",
}

# Candidate (lat, lon) variable/coord names to try, in order, per store. QPESUMS
# uses a regular 0.0125deg lat/lon grid; STA_H8's lat/lon lookup table variable
# name is store-specific and should be added via `latlon_names` if it differs
# (see docs/rainpro_dataset.md: "STA_H8_Plt_IR_for_glbdisplay 有逐像素經緯度查找表").
DEFAULT_LATLON_CANDIDATES = [("lat", "lon"), ("XLAT", "XLONG"), ("latitude", "longitude")]


def _is_zarr_store(path: str) -> bool:
    """A zarr v3 group root has a `zarr.json` directly under it; a raw STA_H8
    directory tree (nested `YYYY/MM/DD/.../*.btp`) never does."""
    return os.path.isfile(os.path.join(path, "zarr.json"))


def split_sta_h8_paths(data_root_value: str) -> list[str]:
    """`data_root["sta_h8"]` may be a single path or a comma-separated list of
    them -- the latter for when `scripts/compress_sta_h8_taiwan.py --freq
    quarter` (or month/day) split one year across several stores (e.g. to fit
    each under a different filesystem's quota, see that script's docstring).
    `dict[str, str]` stays the type (no `list[str]` value) so this doesn't
    collide with the `data_root` dict-merging footgun already documented in
    rainpro8.yml (jsonargparse deep-merges dict-typed CLI params; a
    list-typed value would just make that worse, not better)."""
    return [p.strip() for p in data_root_value.split(",") if p.strip()]


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
        frame_cache_size: int = 64,
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
        # Per-worker LRU cache of *pre-regrid* (store_key, raw_name, resolved
        # timestamp) -> raw (masked) ndarray, keyed on the timestamp actually
        # resolved by `.sel(..., method="nearest")` rather than the query time,
        # so two samples whose offsets snap to the same underlying frame share
        # one disk read. Deliberately caches before regridding (not after):
        # `jitter_km` randomizes `dst_lat`/`dst_lon` per sample, so the
        # regridded result differs per sample even when the source frame is
        # identical -- only the load+missing-mask step is safe to reuse. Most
        # valuable for sources whose offsets overlap heavily between adjacent
        # samples (radar_4km's 7 offsets @ 10 min, satellite_8km's hourly
        # frames reused by every 10-min target sample within that hour) and,
        # even for a single sample, whenever `variables`/`variables_3d` shares
        # a raw store access across bands/levels that live in one file (e.g.
        # `sta_h8_raw`'s raw `.btp` reader) -- see `rainpro/data/sta_h8_raw.py`
        # and `scripts/compress_sta_h8_taiwan.py` for the I/O cost this avoids.
        self._frame_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._frame_cache_size = frame_cache_size

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
            if store_key == "sta_h8":
                paths = split_sta_h8_paths(path)
                zarr_paths = [p for p in paths if _is_zarr_store(p)]
                raw_paths = [p for p in paths if not _is_zarr_store(p)]
                if zarr_paths and raw_paths:
                    raise ValueError(
                        f"data_root['sta_h8'] mixes converted zarr store(s) {zarr_paths} with "
                        f"raw directory path(s) {raw_paths} -- must be all one or the other"
                    )
                if raw_paths:
                    # Not (yet) converted by scripts/compress_sta_h8_taiwan.py --
                    # only a single raw directory root is supported (there's no
                    # natural way to "concat" two overlapping raw directory
                    # trees the way there is for disjoint-in-time zarr stores
                    # below). Falls back to reading raw .btp files directly and
                    # on demand, via a lazy dask-backed xr.Dataset with the same
                    # `.sel(...)` surface a real zarr store would have -- a lot
                    # slower (full uncropped, uncached per-file reads off a
                    # shared filesystem) -- see that script's docstring and
                    # `rainpro/data/sta_h8_raw.py`'s.
                    if len(raw_paths) != 1:
                        raise ValueError(
                            f"data_root['sta_h8'] has {len(raw_paths)} raw (non-zarr) directory "
                            f"paths {raw_paths}; only a single raw directory is supported -- "
                            f"convert with scripts/compress_sta_h8_taiwan.py first if you need "
                            f"to combine multiple sources"
                        )
                    latlon_path = self.data_root.get("sta_h8_latlon", sta_h8_raw.DEFAULT_LATLON_PATH)
                    self._datasets[store_key] = sta_h8_raw.open_sta_h8_raw(raw_paths[0], latlon_path)
                else:
                    # scripts/compress_sta_h8_taiwan.py's stores attempt
                    # consolidation but don't guarantee it (best-effort) --
                    # consolidated=False always works, and unconsolidated open
                    # is only marginally slower for the handful of arrays (9
                    # bands + time/lat/lon) each of these stores has. Multiple
                    # stores (e.g. one per quarter, possibly on different
                    # filesystems -- see that script's `--freq quarter`) are
                    # concatenated along time into one lazy Dataset; `sortby`
                    # guarantees monotonic time regardless of the order the
                    # paths were listed in, which `.sel(..., method="nearest")`
                    # relies on.
                    #
                    # `data_vars="minimal"` is NOT optional here: "lat"/"lon"
                    # are plain (y, x) data variables in this store (no "time"
                    # dim, and nothing marks them as coords -- see
                    # scripts/compress_sta_h8_taiwan.py's `create_array` calls),
                    # so `xr.concat`'s default `data_vars="all"` broadcasts
                    # them along the NEW "time" dim too, i.e. duplicates each
                    # ~6 MB (y, x) array once per timestep (thousands of times)
                    # instead of keeping the one (y, x) array every quarter
                    # already shares (same crop box) -- verified this OOM-kills
                    # a concat of just 2 small test stores. "minimal" only
                    # concatenates variables that actually vary along "time"
                    # (the 9 bands), taking lat/lon from the first dataset as-is.
                    datasets = [xr.open_zarr(p, consolidated=False) for p in zarr_paths]
                    ds = (
                        xr.concat(datasets, dim="time", data_vars="minimal", coords="minimal")
                        if len(datasets) > 1
                        else datasets[0]
                    )
                    self._datasets[store_key] = ds.sortby("time") if len(datasets) > 1 else ds
            else:
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
                    ds,
                    store_key,
                    regridder,
                    spec,
                    init_time,
                    offset,
                    dst_lat,
                    dst_lon,
                    keep_nan=is_target,
                    normalize=not is_target,
                )
                for offset in spec.offsets_min
            ]
            # (T, C, H, W). GT is raw QPESUMS max dBZ (see
            # docs/rainpro_tw_implementation_notes.md) -- no Marshall-Palmer
            # conversion in the data path; `rainpro.data.marshall_palmer` is
            # for post-hoc relabeling only (e.g. `rainpro.metrics.probabilistic.CRPS`).
            arr = np.stack(frames, axis=0)

            sample[name] = torch.from_numpy(arr).float()

        return sample

    def _load_var(
        self, store_key: str, frame_ds: xr.Dataset, raw_name: str, is_static: bool
    ) -> np.ndarray:
        """Raw (unmasked) ndarray for one variable of one resolved frame, via
        `self._frame_cache` (see its docstring in `__init__`). Missing-value
        masking is deliberately NOT done here -- it stays in the caller, applied
        fresh on every call, since it's cheap (pure in-memory numpy) and, unlike
        the raw load, isn't safe to treat as invariant across callers in general.
        """
        if is_static:
            cache_key = (store_key, raw_name, None)
        else:
            # Keyed by the timestamp `.sel(..., method="nearest")` actually
            # resolved to, not the query time -- two different query offsets
            # that snap to the same underlying frame must hit the same cache
            # entry. `.item()` on a 0-d datetime64 array gives a hashable int.
            cache_key = (store_key, raw_name, frame_ds["time"].values.item())

        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            self._frame_cache.move_to_end(cache_key)
            return cached

        data = np.asarray(frame_ds[raw_name].values, dtype=np.float32)
        self._frame_cache[cache_key] = data
        if len(self._frame_cache) > self._frame_cache_size:
            self._frame_cache.popitem(last=False)
        return data

    def _read_frame(
        self,
        ds: xr.Dataset,
        store_key: str,
        regridder: NearestNeighborRegridder,
        spec: SourceSpec,
        init_time: np.datetime64,
        offset_min: int,
        dst_lat: np.ndarray,
        dst_lon: np.ndarray,
        keep_nan: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Returns (C, H, W) for one timestep of one source.

        `normalize=False` (used for `target_2km`) bypasses `minmax_normalize`
        entirely, regardless of whether the variable name happens to collide
        with a `DEFAULT_NORM_BOUNDS`/`norm_bounds` key (`target_2km`'s
        variable is `max_dbz`, which *is* in `DEFAULT_NORM_BOUNDS` for the
        4km/8km radar input tiers) -- the target must stay in raw physical
        units since it's compared against literal dBZ thresholds throughout
        the loss/metrics pipeline, not fed to the network as a normalized
        input feature. Without this, the target was silently squashed to
        [0, 1] here and then (before this fix removed the mm/h conversion
        entirely) run through `dbz_to_mmh` a second time on that already-
        normalized value -- see `docs/rainpro_tw_implementation_notes.md`.
        """
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
            data = self._load_var(store_key, frame_ds, raw_name, is_static)
            data = _mask_missing(data, spec.missing_values)
            regridded = regridder(data, dst_lat, dst_lon, fill_value=np.nan)
            if normalize:
                regridded = minmax_normalize(regridded, self.norm_bounds.get(var))
            channels.append(regridded)

        for var in spec.variables_3d:
            raw_name = self.variable_aliases.get(var, var)
            data = self._load_var(store_key, frame_ds, raw_name, is_static)  # (level, y, x)
            data = data[list(spec.levels)]
            data = _mask_missing(data, spec.missing_values)
            regridded = regridder(data, dst_lat, dst_lon, fill_value=np.nan)
            if normalize:
                regridded = minmax_normalize(regridded, self.norm_bounds.get(var))
            channels.extend(regridded)

        out = np.stack(channels, axis=0)
        if not keep_nan:
            out = np.where(np.isnan(out), self.fill_value, out)
        return out


def _mask_missing(data: np.ndarray, missing_values: Sequence[float]) -> np.ndarray:
    """Replace a source's raw missing-value sentinels (e.g. QPESUMS' -999/-99,
    see `rainpro8_sources.QPESUMS_MISSING_VALUES`) with NaN, in place of the
    literal flag value, before any clipping/normalization happens."""
    if not missing_values:
        return data
    mask = np.zeros(data.shape, dtype=bool)
    for value in missing_values:
        mask |= np.isclose(data, value)
    return np.where(mask, np.nan, data)


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
