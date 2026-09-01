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

import numpy as np
from scipy.spatial import cKDTree

KM_PER_DEG_LAT = 111.32


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

    def __call__(
        self,
        src_data: np.ndarray,
        dst_lat: np.ndarray,
        dst_lon: np.ndarray,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """`src_data` has shape (..., *src_shape); returns shape (..., *dst_lat.shape)."""
        query = np.stack([dst_lat.ravel(), dst_lon.ravel() * self.lon_scale], axis=-1)
        dist_deg, idx = self.tree.query(query)
        dist_km = dist_deg * KM_PER_DEG_LAT

        flat = src_data.reshape(*src_data.shape[: -len(self.src_shape)], -1)
        out = flat[..., idx]
        out = out.reshape(*out.shape[:-1], *dst_lat.shape)

        invalid = (dist_km > self.max_dist_km).reshape(dst_lat.shape)
        out = np.where(invalid, fill_value, out)
        return out
