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
from rainpro.data.regrid import NearestNeighborRegridder, RegridMapping, target_grid
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


def _open_store(path: str, consolidated: bool) -> xr.Dataset:
    """`chunks=None` is the point: it opens the store through xarray's own lazy
    indexing instead of wrapping every array in dask.

    Training reads are small, explicitly indexed and synchronous -- 62 frames a
    sample, one chunk each -- and dask charged ~24 ms of per-chunk scheduler and
    graph-construction overhead on top of ~0.6 ms of real read+decompress.
    Measured on the real stores (`scripts/profile_training_pipeline.py
    --sections readpath`), reading 36 QPESUMS frames:

        xarray + dask            0.5685 s
        xarray, chunks=None      0.0224 s   (25.4x)
        zarr array directly      0.0229 s   (24.9x)

    That `chunks=None` ties reading the zarr arrays directly is what says the
    whole difference was dask rather than anything in the storage layer -- and
    a page-cache test (re-running identical reads) had already ruled out
    filesystem latency, while the cost being flat in *chunks* rather than bytes
    ruled out decompression.

    Dask also spawns a thread pool sized by `os.cpu_count()`, which on a SLURM
    node reports the whole machine rather than the cgroup's allocation: with
    `num_workers=11` that was ~1000 threads contending for 12 cores, the likely
    source of the multi-second p95 stalls seen in a real fit.
    """
    return xr.open_zarr(path, consolidated=consolidated, chunks=None)


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


class _StoreHandle:
    """One logical store, backed by one or more zarr datasets.

    Exists to keep two things that used to be in tension:

    * The stores are opened with `chunks=None`, i.e. NOT dask-backed. Dask was
      costing ~24 ms of scheduler and graph-construction overhead per chunk
      against ~0.6 ms of actual read+decompress -- 25x on the QPESUMS reads
      that dominate a sample (measured, `scripts/profile_training_pipeline.py
      --sections readpath`). Every read here is a small, explicitly indexed,
      synchronous fetch, which is exactly the shape dask is worst at.
    * STA_H8 can be split across several stores (one per quarter, possibly on
      different filesystems -- see `scripts/compress_sta_h8_taiwan.py --freq
      quarter`, needed because /home and /work are 100 GB each). That used to
      be `xr.concat(...).sortby("time")`, which is lazy only while the arrays
      are dask-backed: without dask it would call `np.concatenate` and pull all
      ~121 GB into memory on the first `_get_store`.

    So instead of concatenating the data, this concatenates only the *time
    index* and remembers, per global position, which dataset owns it and where.
    Reads are grouped by owner and issued one `isel` per dataset, which keeps
    `_load_frames`'s batched-read property intact. A 36-offset window spans 6
    hours and so normally sits inside one quarter; near a boundary it simply
    becomes two reads instead of one.
    """

    def __init__(self, datasets: list[xr.Dataset]):
        if not datasets:
            raise ValueError("_StoreHandle needs at least one dataset")
        self.datasets = datasets
        # Variable names, dims and lat/lon are identical across the stores of
        # one source (same crop box, same bands), so inspection can use any.
        self.primary = datasets[0]

        if len(datasets) == 1:
            index = datasets[0].indexes.get("time")
            self.time_index = index if index is not None else pd.DatetimeIndex([])
            self._owner = None  # single-store fast path: position == local index
            self._local = None
            return

        times, owners, locals_ = [], [], []
        for store_id, ds in enumerate(datasets):
            index = ds.indexes["time"]
            times.append(np.asarray(index.values))
            owners.append(np.full(len(index), store_id, dtype=np.int32))
            locals_.append(np.arange(len(index), dtype=np.int64))
        times = np.concatenate(times)
        # Stable sort so `.get_indexer(..., method="nearest")` -- which requires
        # a monotonic index -- works regardless of the order the paths were
        # listed in, exactly as the old `.sortby("time")` guaranteed.
        order = np.argsort(times, kind="stable")
        self.time_index = pd.DatetimeIndex(times[order])
        self._owner = np.concatenate(owners)[order]
        self._local = np.concatenate(locals_)[order]

    def read(self, raw_name: str, positions: np.ndarray | None) -> np.ndarray:
        """`(len(positions), ...)` for one variable, or the whole array when
        `positions is None` (a variable with no time dimension)."""
        if positions is None:
            return np.asarray(self.primary[raw_name].values, dtype=np.float32)

        positions = np.asarray(positions)
        if self._owner is None:
            return np.asarray(
                self.datasets[0][raw_name].isel(time=positions.tolist()).values,
                dtype=np.float32,
            )

        owners = self._owner[positions]
        local = self._local[positions]
        out: np.ndarray | None = None
        for store_id in np.unique(owners):
            slots = np.flatnonzero(owners == store_id)
            # Ascending within the read: zarr's orthogonal indexing is happiest
            # with sorted selections, and scattering back through `slots[order]`
            # keeps the caller's ordering exact.
            order = np.argsort(local[slots], kind="stable")
            block = np.asarray(
                self.datasets[store_id][raw_name]
                .isel(time=local[slots][order].tolist())
                .values,
                dtype=np.float32,
            )
            if out is None:
                out = np.empty((len(positions), *block.shape[1:]), dtype=np.float32)
            out[slots[order]] = block
        assert out is not None  # `positions` is non-empty by construction
        return out

    def close(self) -> None:
        for ds in self.datasets:
            ds.close()


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
        for handle in self._datasets.values():
            handle.close()
        self._datasets = {}

    # -- lazy zarr access -------------------------------------------------

    def _get_store(self, store_key: str) -> "_StoreHandle":
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
                    self._datasets[store_key] = _StoreHandle(
                        [sta_h8_raw.open_sta_h8_raw(raw_paths[0], latlon_path)]
                    )
                else:
                    # scripts/compress_sta_h8_taiwan.py's stores attempt
                    # consolidation but don't guarantee it (best-effort) --
                    # consolidated=False always works, and unconsolidated open
                    # is only marginally slower for the handful of arrays (9
                    # bands + time/lat/lon) each of these stores has.
                    #
                    # Multiple stores (one per quarter, possibly on different
                    # filesystems -- see that script's `--freq quarter`) are
                    # NOT concatenated. `_StoreHandle` merges their time indices
                    # and routes each read to the owning store instead; see its
                    # docstring for why concatenating is unsafe once the arrays
                    # stop being dask-backed.
                    self._datasets[store_key] = _StoreHandle(
                        [_open_store(p, consolidated=False) for p in zarr_paths]
                    )
            else:
                self._datasets[store_key] = _StoreHandle(
                    [_open_store(path, consolidated=True)]
                )
        return self._datasets[store_key]

    def _get_regridder(self, store_key: str) -> NearestNeighborRegridder:
        if store_key not in self._regridders:
            handle = self._get_store(store_key)
            candidates = (
                [self.latlon_names[store_key]] + DEFAULT_LATLON_CANDIDATES
                if store_key in self.latlon_names
                else DEFAULT_LATLON_CANDIDATES
            )
            lat, lon = _extract_latlon(handle.primary, candidates)
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
            handle = self._get_store(store_key)
            regridder = self._get_regridder(store_key)
            dst_lat, dst_lon = target_grid(center_lat, center_lon, spec.size_km, spec.resolution_km)
            # Computed once per (sample, source), not once per offset: every
            # offset of one source regrids onto the SAME destination grid
            # (`dst_lat`/`dst_lon` don't vary with `offset_min`), so the
            # nearest-neighbor `cKDTree.query()` behind it is pure repeated
            # work otherwise -- up to 36x over for target_2km's offsets, 7x
            # for radar_4km's, 2x for satellite_8km's. See `RegridMapping`'s
            # docstring in rainpro/data/regrid.py.
            mapping = regridder.prepare(dst_lat, dst_lon)

            is_target = name == "target_2km"
            # (T, C, H, W). GT is raw QPESUMS max dBZ (see
            # docs/rainpro_tw_implementation_notes.md) -- no Marshall-Palmer
            # conversion in the data path; `rainpro.data.marshall_palmer` is
            # for post-hoc relabeling only (e.g. `rainpro.metrics.probabilistic.CRPS`).
            arr = self._read_source(
                handle,
                store_key,
                regridder,
                mapping,
                spec,
                init_time,
                keep_nan=is_target,
                normalize=not is_target,
            )

            sample[name] = torch.from_numpy(arr).float()

        return sample

    def _cache_put(self, key: tuple, value: np.ndarray) -> None:
        self._frame_cache[key] = value
        self._frame_cache.move_to_end(key)
        if len(self._frame_cache) > self._frame_cache_size:
            self._frame_cache.popitem(last=False)

    def _load_frames(
        self, store_key: str, handle: "_StoreHandle", raw_name: str, positions: np.ndarray | None
    ) -> list[np.ndarray]:
        """Source-resolution arrays for one variable at `positions` (integer,
        sorted, unique indices into the store's time axis), or a one-element
        list when `positions is None` (a variable with no time dimension).

        Every cache-missing position is fetched in ONE `isel`, so dask/zarr
        can issue those chunk reads concurrently, rather than one blocking
        read per timestep. That round-trip count is the point: `target_2km`
        alone asks for 36 offsets, and against a shared network filesystem
        (this project's stores live on HFS, not node-local NVMe) it's the
        per-chunk latency, not the bandwidth, that dominates -- 36 serialized
        round-trips per sample per variable before this.

        Caching is per (store, variable, position). It's near-worthless for
        the shuffled train split -- two of one worker's samples landing within
        each other's offset window is a <0.1% event there -- but a large win
        for val/test, which iterate in time order (`shuffle=False`), where
        consecutive `target_2km` samples overlap in 35 of their 36 offsets.
        """
        if positions is None:
            key = (store_key, raw_name, None)
            cached = self._frame_cache.get(key)
            if cached is not None:
                self._frame_cache.move_to_end(key)
                return [cached]
            data = handle.read(raw_name, None)
            self._cache_put(key, data)
            return [data]

        frames: list[np.ndarray | None] = [None] * len(positions)
        missing_pos: list[int] = []
        missing_slots: list[int] = []
        for i, pos in enumerate(positions):
            key = (store_key, raw_name, int(pos))
            cached = self._frame_cache.get(key)
            if cached is None:
                missing_pos.append(int(pos))
                missing_slots.append(i)
            else:
                self._frame_cache.move_to_end(key)
                frames[i] = cached

        if missing_pos:
            block = handle.read(raw_name, np.asarray(missing_pos))
            for j, slot in enumerate(missing_slots):
                # `block[j]` is a view onto the whole fetched block, so caching
                # it as-is would keep every *other* frame in that block alive
                # too -- the LRU's entry count would stop bounding real memory.
                frame = block[j].copy()
                frames[slot] = frame
                self._cache_put((store_key, raw_name, missing_pos[j]), frame)

        return frames  # type: ignore[return-value]

    def _read_source(
        self,
        handle: "_StoreHandle",
        store_key: str,
        regridder: NearestNeighborRegridder,
        mapping: RegridMapping,
        spec: SourceSpec,
        init_time: np.datetime64,
        keep_nan: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Returns (T, C, H, W) for one source: every offset in
        `spec.offsets_min`, every channel, regridded onto `mapping`.

        All of a variable's offsets are resolved and fetched together (see
        `_load_frames`) instead of one blocking `.sel(time=..., method=
        "nearest")` per offset. Offsets with no timestep within tolerance
        (e.g. a sensor outage) come back as -1 from `get_indexer` and are left
        as NaN here, matching the per-offset `except KeyError` branch this
        replaced.

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
        raw_names = [
            self.variable_aliases.get(v, v)
            for v in list(spec.variables) + list(spec.variables_3d)
        ]
        # A source is "static" if none of its variables actually carry a time
        # dimension in the store, even if the store also holds other,
        # time-varying variables. Static sources skip time selection entirely.
        is_static = not any("time" in handle.primary[n].dims for n in raw_names)

        n_times = len(spec.offsets_min)
        out = np.full((n_times, spec.channels, *mapping.dst_shape), np.nan, dtype=np.float32)

        if is_static:
            positions = None
            # every offset reads the same (only) frame
            slots = np.zeros(n_times, dtype=int)
        else:
            query_times = pd.DatetimeIndex(
                [pd.Timestamp(init_time + np.timedelta64(o, "m")) for o in spec.offsets_min]
            )
            found_at = handle.time_index.get_indexer(
                query_times, method="nearest", tolerance=_tolerance(spec)
            )
            within = found_at >= 0  # -1 == nothing within tolerance
            if not within.any():
                return out if keep_nan else np.where(np.isnan(out), self.fill_value, out)
            positions, inverse = np.unique(found_at[within], return_inverse=True)
            slots = np.full(n_times, -1, dtype=int)
            slots[within] = inverse

        per_var_frames = [
            self._load_frames(store_key, handle, raw_name, positions)
            for raw_name in raw_names
        ]
        n_2d = len(spec.variables)
        n_levels = len(spec.levels)

        for t_idx, slot in enumerate(slots):
            if slot < 0:
                continue  # no timestep within tolerance -- stays NaN
            channel = 0
            for var_idx, var in enumerate(spec.variables):
                data = _fill_no_echo(
                    per_var_frames[var_idx][slot], spec.no_echo_values, spec.no_echo_fill
                )
                data = _mask_missing(data, spec.missing_values)
                regridded = regridder.apply(data, mapping, fill_value=np.nan)
                if normalize:
                    regridded = minmax_normalize(regridded, self.norm_bounds.get(var))
                out[t_idx, channel] = regridded
                channel += 1

            for var_idx, var in enumerate(spec.variables_3d):
                data = per_var_frames[n_2d + var_idx][slot]  # (level, y, x)
                data = _fill_no_echo(
                    data[list(spec.levels)], spec.no_echo_values, spec.no_echo_fill
                )
                data = _mask_missing(data, spec.missing_values)
                regridded = regridder.apply(data, mapping, fill_value=np.nan)
                if normalize:
                    regridded = minmax_normalize(regridded, self.norm_bounds.get(var))
                out[t_idx, channel : channel + n_levels] = regridded
                channel += n_levels

        if not keep_nan:
            out = np.where(np.isnan(out), self.fill_value, out)
        return out


def _fill_no_echo(data: np.ndarray, no_echo_values: Sequence[float], fill: float) -> np.ndarray:
    """Replace "observed, nothing here" sentinels (QPESUMS' -99) with a real
    low value rather than NaN.

    These are the opposite of `_mask_missing`'s values despite looking the
    same: they're the negative examples -- ~98% of a QPESUMS frame -- and
    masking them out is what left the model with nothing teaching it where
    rain ISN'T. See `rainpro8_sources.QPESUMS_NO_ECHO_VALUES`."""
    if not no_echo_values:
        return data
    mask = np.zeros(data.shape, dtype=bool)
    for value in no_echo_values:
        mask |= np.isclose(data, value)
    return np.where(mask, fill, data)


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
