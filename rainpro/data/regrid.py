"""Generic lat/lon nearest-neighbor regridding onto a local equirectangular canvas.

Taiwan sources live on different native grids (QPESUMS regular lat/lon,
STA_H8 Himawari LCC with a per-pixel lookup table). Rather than hard-coding
each projection's parameters, every source is regridded via nearest-neighbor
lookup against its own (lat, lon) coordinate arrays onto a common target
lat/lon grid -- this only requires that each store expose per-pixel
coordinates, which all of the above already do (see `docs/rainpro_dataset.md`).

This is a lightweight approximation (local equirectangular projection, plain
nearest-neighbor) intended to get a working pipeline; production use may want
bilinear/conservative regridding (e.g. via `pyresample`) instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

KM_PER_DEG_LAT = 111.32


@dataclass(frozen=True)
class RegridMapping:
    """Precomputed nearest-neighbor mapping from one regridder's source grid to
    one destination grid -- everything `NearestNeighborRegridder.__call__` has
    to do a `cKDTree.query()` for. Building it is the expensive part (one
    query per destination pixel); applying it to a `src_data` array afterward
    is just fancy indexing. Compute once via `NearestNeighborRegridder.
    prepare()` and reuse via `.apply()` for every array that shares the same
    (dst_lat, dst_lon) -- e.g. RainPro8Dataset regrids the same source onto
    the same per-sample destination grid once per timestep offset (up to 36x
    for target_2km) with `dst_lat`/`dst_lon` identical across every one of
    those calls; querying the tree fresh each time was pure waste."""

    idx: np.ndarray  # flat index into the source grid, one per dst pixel
    invalid: np.ndarray  # dst_shape bool, True where no source pixel is within max_dist_km
    dst_shape: tuple[int, ...]


def target_grid(
    center_lat: float,
    center_lon: float,
    size_km: float,
    resolution_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Regular lat/lon grid of `size_km` x `size_km`, `resolution_km`/px, centered
    at (center_lat, center_lon). Returns (lat, lon), each of shape (size_px, size_px).
    """
    size_px = round(size_km / resolution_km)
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.deg2rad(center_lat))

    offsets_km = (np.arange(size_px) - (size_px - 1) / 2) * resolution_km
    lat = center_lat + offsets_km / KM_PER_DEG_LAT
    lon = center_lon + offsets_km / km_per_deg_lon

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return lat_grid, lon_grid


class NearestNeighborRegridder:
    """Caches the KDTree over a source's (lat, lon) points for repeated queries
    against different destination grids (e.g. per-sample spatial jitter).

    Longitude is rescaled by cos(ref_lat) before building the tree so that
    Euclidean distance in (lat, scaled-lon) degrees approximates true great-circle
    distance near `ref_lat`, avoiding anisotropic distortion at Taiwan's latitude.
    """

    def __init__(
        self,
        src_lat: np.ndarray,
        src_lon: np.ndarray,
        ref_lat: float,
        max_dist_km: float = 15.0,
    ):
        self.src_shape = src_lat.shape
        self.lon_scale = np.cos(np.deg2rad(ref_lat))
        points = np.stack([src_lat.ravel(), src_lon.ravel() * self.lon_scale], axis=-1)
        self.tree = cKDTree(points)
        self.max_dist_km = max_dist_km

    def prepare(self, dst_lat: np.ndarray, dst_lon: np.ndarray) -> RegridMapping:
        """The expensive half of regridding (one `cKDTree.query()` per
        destination pixel) -- see `RegridMapping`'s docstring for why this is
        split out from `apply()`/`__call__()`."""
        query = np.stack([dst_lat.ravel(), dst_lon.ravel() * self.lon_scale], axis=-1)
        dist_deg, idx = self.tree.query(query)
        dist_km = dist_deg * KM_PER_DEG_LAT
        invalid = (dist_km > self.max_dist_km).reshape(dst_lat.shape)
        return RegridMapping(idx=idx, invalid=invalid, dst_shape=dst_lat.shape)

    def apply(
        self, src_data: np.ndarray, mapping: RegridMapping, fill_value: float = np.nan
    ) -> np.ndarray:
        """`src_data` has shape (..., *src_shape); returns shape (..., *mapping.dst_shape)."""
        flat = src_data.reshape(*src_data.shape[: -len(self.src_shape)], -1)
        out = flat[..., mapping.idx]
        out = out.reshape(*out.shape[:-1], *mapping.dst_shape)
        out = np.where(mapping.invalid, fill_value, out)
        return out

    def __call__(
        self,
        src_data: np.ndarray,
        dst_lat: np.ndarray,
        dst_lon: np.ndarray,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """One-shot convenience path -- recomputes the mapping every call.
        Prefer `prepare()` once + `apply()` per array when regridding several
        arrays onto the SAME (dst_lat, dst_lon) (e.g. multiple timesteps of
        one sample's one source, see RainPro8Dataset.__getitem__)."""
        return self.apply(src_data, self.prepare(dst_lat, dst_lon), fill_value)
