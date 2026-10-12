"""Area aggregation (`rainpro.data.regrid.aggregate`) on hand-checkable grids."""

import numpy as np
import pytest

from rainpro.data.regrid import aggregate, prepare_aggregation, target_grid

# A 4 x 4 source grid of 1 km pixels around (0 N, 0 E) aggregated onto a
# 2 x 2 canvas of 2 km cells: each cell holds exactly one 2 x 2 pixel block.
KM_DEG = 1 / 111.32
OFFSETS = (np.arange(4) - 1.5) * KM_DEG
SRC_LAT, SRC_LON = np.meshgrid(OFFSETS, OFFSETS, indexing="ij")
MAPPING = prepare_aggregation(SRC_LAT, SRC_LON, 0.0, 0.0, size_km=4, resolution_km=2)


def blocks(values):
    """4 x 4 field whose 2 x 2 blocks (south-west first, row-major) carry the
    given four lists of pixel values."""
    field = np.empty((4, 4))
    for (r, c), vals in zip([(0, 0), (0, 2), (2, 0), (2, 2)], values):
        field[r:r + 2, c:c + 2] = np.reshape(vals, (2, 2))
    return field


def test_partition_and_alignment_with_target_grid():
    assert MAPPING.dst_shape == (2, 2)
    assert MAPPING.keep.all()  # every source pixel lands in some cell
    assert np.bincount(MAPPING.cell, minlength=4).tolist() == [4, 4, 4, 4]
    # cell (0, 0) is the south-west one, matching target_grid's lat-ascending rows
    lat, lon = target_grid(0.0, 0.0, 4, 2)
    assert lat[0, 0] < lat[1, 0] and lon[0, 0] < lon[0, 1]


def test_mean_zmean_max():
    field = blocks([[0, 0, 40, 40], [10, 10, 10, 10], [0, 0, 0, 0], [20, 30, 40, 50]])
    mean = aggregate(field, MAPPING, "mean")
    zmean = aggregate(field, MAPPING, "zmean")
    vmax = aggregate(field, MAPPING, "max")

    assert mean[0, 0] == pytest.approx(20.0)
    # half 0 dBZ (Z = 1), half 40 dBZ (Z = 1e4): 10 log10(5000.5)
    assert zmean[0, 0] == pytest.approx(10 * np.log10((1 + 1e4) / 2), abs=1e-4)
    assert zmean[0, 1] == pytest.approx(10.0, abs=1e-4)  # constant cell is unchanged
    assert vmax[1, 1] == pytest.approx(50.0)
    assert zmean[0, 0] > mean[0, 0]  # averaging dBZ under-weights strong echo


def test_nan_pixels_ignored_and_empty_cell_is_nan():
    field = blocks([[np.nan, 30, 30, 30], [np.nan] * 4, [5, 5, 5, 5], [0, 0, 0, 0]])
    mean = aggregate(field, MAPPING, "mean")
    assert mean[0, 0] == pytest.approx(30.0)
    assert np.isnan(mean[0, 1])


def test_leading_dims_reduced_independently():
    a = blocks([[1] * 4, [2] * 4, [3] * 4, [4] * 4])
    stacked = np.stack([a, a + 10])  # e.g. two levels
    out = aggregate(stacked, MAPPING, "mean")
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out[1], out[0] + 10)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        aggregate(np.zeros((4, 4)), MAPPING, "median")
