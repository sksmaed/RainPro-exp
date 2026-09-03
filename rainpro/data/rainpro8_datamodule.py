"""LightningDataModule for RainPro-8 (Taiwan): QPESUMS + STA_H8 obs-only baseline,
with GFS available as an optional input behind `include_gfs` (see `rainpro8_sources.py`).
"""

from __future__ import annotations

import datetime
from typing import Literal

import pandas as pd
import xarray as xr
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from rainpro.data.rainpro8_dataset import RainPro8Dataset
from rainpro.data.rainpro8_sources import GFS_ANALYSIS_VARIABLES, SourceSpec, build_taiwan_sources


def cycle_split(
    start: datetime.datetime,
    end: datetime.datetime,
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


class RainPro8DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: dict[str, str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
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
        norm_bounds: dict[str, tuple[float, float]] | None = None,
        variable_aliases: dict[str, str] | None = None,
        latlon_names: dict[str, tuple[str, str]] | None = None,
        gfs_variables: tuple[str, ...] = GFS_ANALYSIS_VARIABLES,
        gfs_forecast_variables: tuple[str, ...] = ("PRATE",),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.include_gfs = include_gfs
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.train_jitter_km = train_jitter_km
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.norm_bounds = norm_bounds
        self.variable_aliases = variable_aliases
        self.latlon_names = latlon_names

        self.sources: dict[str, SourceSpec] = build_taiwan_sources(
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
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size if split == "train" else self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=split == "train",
            drop_last=split == "train",
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader("test")
