"""`manifest_splits`: role assignment, the exact role gap, and train-only
low-echo filtering, on a small manifest with missing timesteps."""

import numpy as np
import pandas as pd
import pytest

from rainpro.data.rainpro8_datamodule import manifest_splits


def min_gap_hours(a, b) -> float:
    a = np.sort(np.array(a, dtype="datetime64[m]"))
    b = np.sort(np.array(b, dtype="datetime64[m]"))
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return min(np.abs(a - b[i - 1]).min(), np.abs(b[i] - a).min()) / np.timedelta64(1, "h")


@pytest.fixture
def manifest(tmp_path):
    rng = np.random.default_rng(0)
    times = pd.date_range("2021-06-01", "2021-06-12 23:50", freq="10min")
    times = times[rng.random(len(times)) > 0.3]  # holes, as in the real pool
    day = (times - times[0]).days
    partition = np.select([day < 3, day < 6, day < 9], ["fold0", "fold1", "test"], default="fold2")
    path = tmp_path / "cv.csv"
    pd.DataFrame({"init_time": times, "partition": partition,
                  "keep": rng.random(len(times)) > 0.5}).to_csv(path, index=False)
    return path, dict(zip(times, partition))


@pytest.mark.parametrize("fold", [0, 1, 2])
def test_roles_gap_and_filter(manifest, fold):
    path, partition = manifest
    df = pd.read_csv(path, parse_dates=["init_time"])
    keep = dict(zip(df["init_time"], df["keep"]))
    splits, counts = manifest_splits(str(path), fold, gap_hours=12.0, train_low_echo_filter=True)

    assert all(partition[t] == f"fold{fold}" for t in splits["val"])
    assert all(partition[t] == "test" for t in splits["test"])
    assert all(partition[t] not in (f"fold{fold}", "test") for t in splits["train"])
    assert all(keep[t] for t in splits["train"])  # train is filtered ...
    assert not all(keep[t] for t in splits["val"])  # ... val is not

    assert counts["test"]["used"] == counts["test"]["pool"]  # test is never trimmed
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if splits[a] and splits[b]:
            assert min_gap_hours(splits[a], splits[b]) >= 12.0


def test_unknown_fold_rejected(manifest):
    path, _ = manifest
    with pytest.raises(ValueError, match="fold7"):
        manifest_splits(str(path), 7)
