"""Code is adapted from https://github.com/DeminYu98/DiffCast/blob/main/datasets/dataset_sevir.py."""

import datetime
import os
from typing import Callable, Dict, Sequence

import h5py
import numpy as np
import pandas as pd
import torch

SEVIR_DATA_TYPES = ["vis", "ir069", "ir107", "vil", "lght"]
PREPROCESS_SCALE = {
    "vil": 1 / 255,
}
PREPROCESS_OFFSET = {
    "vil": 0,
}


class SEVIRDataset(torch.utils.data.Dataset):
    """
    Torch Dataset where:
        - each index returns ONE temporal sequence
        - sequences are extracted from events using sliding windows
        - output tensors have shape (H, W, seq_len)
    """

    def __init__(
        self,
        dataset_dir: str,
        data_types: Sequence[str] = SEVIR_DATA_TYPES,
        transforms: Dict[str, Callable] | None = None,
        seq_len: int = 25,
        raw_seq_len: int = 49,
        stride: int = 12,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        verbose: bool = False,
    ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.data_types = data_types
        self.transforms = transforms
        self.seq_len = seq_len
        self.raw_seq_len = raw_seq_len
        self.stride = stride
        self.verbose = verbose

        # Load catalog
        catalog_path = os.path.join(dataset_dir, "CATALOG.csv")
        catalog = pd.read_csv(catalog_path, parse_dates=["time_utc"], low_memory=False)

        # Filter catalog by dates and missing data
        if start_date is not None:
            catalog = catalog[catalog.time_utc > start_date]
        if end_date is not None:
            catalog = catalog[catalog.time_utc <= end_date]
        catalog = catalog[catalog.pct_missing == 0]

        # Keep only events with all requested data types
        imgt = set(self.data_types)
        filtcat = catalog[np.logical_or.reduce([catalog.img_type == i for i in imgt])]
        filtcat = filtcat.groupby("id").filter(
            lambda x: imgt.issubset(set(x["img_type"]))
        )

        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        filtcat = filtcat.groupby("id").filter(lambda x: x.shape[0] == len(imgt))
        self.samples = (
            filtcat.groupby("id")
            .apply(self._df_to_series, include_groups=False)  # Added include groups
            .reset_index(drop=True)
        )

        # Sequence indexing
        self.num_seq_per_event = 1 + (self.raw_seq_len - self.seq_len) // self.stride
        self.total_sequences = len(self.samples) * self.num_seq_per_event

        # Open HDF5 files
        self.sevir_data_dir = os.path.join(dataset_dir, "data")
        self._hdf_files = {}
        self._open_files()

    def _df_to_series(self, df):
        d = {}
        df = df.set_index("img_type")
        for t in self.data_types:
            s = df.loc[t]
            idx = s.file_index if t != "lght" else s.id
            d[f"{t}_filename"] = s.file_name
            d[f"{t}_index"] = idx
        return pd.Series(d)

    def _open_files(self):
        filenames = set()
        for t in self.data_types:
            filenames |= set(self.samples[f"{t}_filename"].values)

        for f in filenames:
            if self.verbose:
                print("Opening", f)
            self._hdf_files[f] = h5py.File(os.path.join(self.sevir_data_dir, f), "r")

    def close(self):
        for f in self._hdf_files.values():
            f.close()
        self._hdf_files = {}

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, index):
        # map flat index → event + sequence
        event_idx = index // self.num_seq_per_event
        seq_idx = index % self.num_seq_per_event

        row = self.samples.iloc[event_idx]
        t0 = seq_idx * self.stride
        t1 = t0 + self.seq_len

        seq = {}

        for t in self.data_types:
            fname = row[f"{t}_filename"]
            idx = row[f"{t}_index"]

            if t == "lght":
                raise NotImplementedError("lght not supported here")
            else:
                # (H, W, T)
                x = self._hdf_files[fname][t][idx, :, :, t0:t1]

            x = torch.from_numpy(x).float()
            x = PREPROCESS_SCALE[t] * x + PREPROCESS_OFFSET[t]

            if self.transforms is not None and t in self.transforms:
                x = self.transforms[t](x)

            seq[t] = x

        return seq
