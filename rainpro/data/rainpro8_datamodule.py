"""LightningDataModule for RainPro-8 (Taiwan): QPESUMS + STA_H8 obs-only baseline,
with STA_H8 and GFS each togglable (`include_satellite`, `include_gfs`; see
`rainpro8_sources.py`) for radar-only / obs-only / obs+GFS ablation arms.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import DataLoader

from rainpro.data import sta_h8_raw
from rainpro.data.rainpro8_dataset import (
    TIME_TOLERANCE,
    RainPro8Dataset,
    _is_zarr_store,
    split_sta_h8_paths,
)
from rainpro.data.rainpro8_sources import GFS_ANALYSIS_VARIABLES, SourceSpec, build_taiwan_sources


def cycle_split(
    start: datetime.datetime | str,
    end: datetime.datetime | str,
    train_days: int,
    val_days: int,
    test_days: int,
    blackout_hours: float,
    freq_minutes: int,
) -> dict[str, list[pd.Timestamp]]:
    """Multi-day-cycle train/val/test split with a blackout buffer at every
    boundary, following the paper (Sec. A.3): repeating (train_days, val_days,
    test_days) cycles, with `blackout_hours` excluded around each boundary to
    avoid leakage between splits (input/lookback and forecast windows can
    otherwise straddle a split boundary)."""
    blackout = pd.Timedelta(hours=blackout_hours)
    freq = pd.Timedelta(minutes=freq_minutes)
    times: dict[str, list[pd.Timestamp]] = {"train": [], "val": [], "test": []}

    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur < end_ts:
        bounds = {
            "train": (cur, cur + pd.Timedelta(days=train_days)),
            "val": (
                cur + pd.Timedelta(days=train_days),
                cur + pd.Timedelta(days=train_days + val_days),
            ),
            "test": (
                cur + pd.Timedelta(days=train_days + val_days),
                cur + pd.Timedelta(days=train_days + val_days + test_days),
            ),
        }
        for split, (seg_start, seg_end) in bounds.items():
            seg_start = seg_start + blackout / 2
            seg_end = min(seg_end, end_ts) - blackout / 2
            if seg_end <= seg_start:
                continue
            times[split].extend(pd.date_range(seg_start, seg_end, freq=freq))

        cur = cur + pd.Timedelta(days=train_days + val_days + test_days)

    return times


def _zarr_written_time_positions(store_path: str, band: str) -> set[int]:
    """Time positions of `band` whose chunk was actually written to a
    scripts/compress_sta_h8_taiwan.py store.

    That script lays out a complete hourly time axis and only writes the
    (time, band) frames that exist and pass validation, leaving the rest at
    the array's NaN fill_value -- so the store's `time` coordinate alone says
    nothing about availability. With its (1, y, x) chunking every frame is
    exactly one chunk, and an unwritten one has no chunk on disk, so listing
    the chunk directory answers this without reading any pixel data (a full
    year is hundreds of GB decompressed)."""
    with open(os.path.join(store_path, band, "zarr.json")) as f:
        meta = json.load(f)
    encoding = meta.get("chunk_key_encoding", {})
    separator = encoding.get("configuration", {}).get("separator", "/")
    chunk_t = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
    if encoding.get("name", "default") != "default" or separator != "/" or chunk_t != 1:
        raise ValueError(
            f"{store_path}/{band}: expected zarr v3 default '/' chunk keys with one "
            f"timestep per chunk (scripts/compress_sta_h8_taiwan.py's layout), got "
            f"chunk_key_encoding={encoding} chunk_shape[0]={chunk_t}"
        )
    chunk_dir = os.path.join(store_path, band, "c")
    if not os.path.isdir(chunk_dir):
        return set()
    return {int(name) for name in os.listdir(chunk_dir) if name.isdigit()}


def sta_h8_availability(path: str) -> pd.Series:
    """Bool Series indexed by the store's own time axis (the one
    `RainPro8Dataset` resolves offsets against): True only where all 9 bands
    have a real frame. A timestep missing any band would otherwise reach the
    model as `fill_value` (0, i.e. 180 K after normalization -- the coldest
    cloud top, not "no data")."""
    if _is_zarr_store(path):
        with xr.open_zarr(path, consolidated=False) as ds:
            times = pd.DatetimeIndex(ds["time"].values)
        valid = np.ones(len(times), dtype=bool)
        for band in sta_h8_raw.BANDS:
            band_written = np.zeros(len(times), dtype=bool)
            band_written[sorted(_zarr_written_time_positions(path, band))] = True
            valid &= band_written
        return pd.Series(valid, index=times)

    # Raw `.btp` tree: `sta_h8_raw.open_sta_h8_raw` puts every timestamp with
    # *any* band on its time axis. A wrong-sized file is read as NaN at load
    # time (`_load_or_nan`), so check size here too -- a stat, not a read.
    file_index, times = sta_h8_raw.scan_files(path)
    expected_bytes = sta_h8_raw.IX * sta_h8_raw.IY * 4
    valid = [
        all(
            (t, band) in file_index and os.path.getsize(file_index[(t, band)]) == expected_bytes
            for band in sta_h8_raw.BANDS
        )
        for t in times
    ]
    return pd.Series(valid, index=pd.DatetimeIndex(times), dtype=bool)


class RainPro8DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: dict[str, str],
        # `str` (e.g. "2021-06-01"), not `datetime.datetime`: jsonargparse has no
        # built-in CLI/YAML parsing for bare `datetime.datetime`-typed params (it
        # falls through to its Path/class-path fallbacks and errors on any plain
        # date string, from --config *or* CLI overrides alike). `cycle_split`
        # converts via `pd.Timestamp`, which accepts ISO date strings directly.
        start_date: str,
        end_date: str,
        include_satellite: bool = True,
        include_gfs: bool = False,
        cycle_train_days: int = 12,
        cycle_val_days: int = 2,
        cycle_test_days: int = 2,
        cycle_blackout_hours: float = 12,
        center_lat: float = 23.7,
        center_lon: float = 121.0,
        train_jitter_km: float = 256,
        batch_size: int = 16,
        eval_batch_size: int | None = None,
        num_workers: int = 8,
        persistent_workers: bool = True,
        prefetch_factor: int | None = 4,
        frame_cache_size: int = 64,
        norm_bounds: dict[str, tuple[float, float]] | None = None,
        variable_aliases: dict[str, str] | None = None,
        latlon_names: dict[str, tuple[str, str]] | None = None,
        gfs_variables: tuple[str, ...] = GFS_ANALYSIS_VARIABLES,
        gfs_forecast_variables: tuple[str, ...] = ("PRATE",),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.include_satellite = include_satellite
        self.include_gfs = include_gfs
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.train_jitter_km = train_jitter_km
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        # persistent_workers=True keeps worker processes (and everything they've
        # lazily built: open zarr/dask handles, KDTree regridders, and each
        # worker's RainPro8Dataset._frame_cache) alive across epochs instead of
        # tearing them down and re-forking from scratch every epoch -- with the
        # raw (non-zarr) STA_H8 fallback path in particular, a fresh worker
        # re-does a full os.walk of the source tree via sta_h8_raw.scan_files()
        # on first access, so without this that scan (and every store's lazy-open
        # cost) repeats every single epoch. Only meaningful when num_workers > 0
        # (torch.utils.data.DataLoader rejects it otherwise).
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.frame_cache_size = frame_cache_size
        self.norm_bounds = norm_bounds
        self.variable_aliases = variable_aliases
        self.latlon_names = latlon_names

        self.sources: dict[str, SourceSpec] = build_taiwan_sources(
            include_satellite=include_satellite,
            include_gfs=include_gfs,
            gfs_variables=gfs_variables,
            gfs_forecast_variables=gfs_forecast_variables,
        )

        target_offsets = self.sources["target_2km"].offsets_min
        target_cadence_min = target_offsets[1] - target_offsets[0]  # 10 min
        self.split_times = cycle_split(
            start_date,
            end_date,
            cycle_train_days,
            cycle_val_days,
            cycle_test_days,
            cycle_blackout_hours,
            freq_minutes=target_cadence_min,
        )

    @property
    def frames_out(self) -> int:
        return self.sources["target_2km"].timesteps

    def setup(self, stage: str | None = None):
        # Restrict candidate init times to those actually present in the QPESUMS
        # target store; per-sample radar-coverage filtering (paper: >=50% for
        # train, allowed lower for val/test) is left to a `coverage` variable in
        # the QPESUMS store if present, since computing it here would require
        # eagerly reading every candidate timestep.
        qpesums_path = self.data_root.get("qpesums")
        if qpesums_path is None:
            return
        with xr.open_zarr(qpesums_path, consolidated=True) as ds:
            available = set(pd.DatetimeIndex(ds["time"].values))
        for split in self.split_times:
            self.split_times[split] = [t for t in self.split_times[split] if t in available]

        # STA_H8's real archive has large fully-missing stretches (e.g. ~2
        # months at the start of 2021, see scripts/inspect_sta_h8_times.py)
        # plus scattered missing hours/bands. Any `satellite_8km` offset that
        # resolves to a missing frame reaches the model as `fill_value`, so
        # drop those init_times. `sta_h8_path` is a comma-separated list
        # (usually length 1) of either raw STA_H8 directory roots or zarr v3
        # stores from scripts/compress_sta_h8_taiwan.py -- same detection and
        # multi-store convention as `RainPro8Dataset._get_store`.
        #
        # Checking the time axis alone is not enough: the zarr stores carry a
        # complete hourly axis with NaN frames wherever no file existed, and
        # the raw reader lists a timestamp if *any* band exists. So each
        # offset is resolved against the full axis exactly as the Dataset
        # does (same nearest/tolerance lookup), then the frame it lands on
        # must have all 9 bands (`sta_h8_availability`).
        sta_h8_path = self.data_root.get("sta_h8")
        if self.include_satellite and sta_h8_path is not None:
            availability = pd.concat(
                [sta_h8_availability(p) for p in split_sta_h8_paths(sta_h8_path)]
            ).sort_index()
            sat_index = availability.index
            sat_valid = availability.to_numpy()
            rank_zero_info(
                f"[STA_H8] {int(sat_valid.sum())}/{len(sat_valid)} timesteps on the store's "
                f"time axis have all {len(sta_h8_raw.BANDS)} bands"
            )
            offsets = self.sources["satellite_8km"].offsets_min
            tolerance = pd.Timedelta(TIME_TOLERANCE["satellite_8km"])
            for split in self.split_times:
                times = pd.DatetimeIndex(self.split_times[split])
                covered = np.ones(len(times), dtype=bool)
                for offset in offsets:
                    query = times + pd.Timedelta(minutes=offset)
                    pos = sat_index.get_indexer(query, method="nearest", tolerance=tolerance)
                    covered &= (pos >= 0) & sat_valid[np.maximum(pos, 0)]
                rank_zero_info(
                    f"[STA_H8] {split}: kept {int(covered.sum())}/{len(times)} init_times "
                    f"with complete satellite input"
                )
                self.split_times[split] = list(times[covered])

    def _dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        dataset = RainPro8Dataset(
            data_root=self.data_root,
            sources=self.sources,
            init_times=self.split_times[split],
            center_lat=self.center_lat,
            center_lon=self.center_lon,
            jitter_km=self.train_jitter_km if split == "train" else 0.0,
            norm_bounds=self.norm_bounds,
            variable_aliases=self.variable_aliases,
            latlon_names=self.latlon_names,
            frame_cache_size=self.frame_cache_size,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size if split == "train" else self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=split == "train",
            drop_last=split == "train",
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader("test")
