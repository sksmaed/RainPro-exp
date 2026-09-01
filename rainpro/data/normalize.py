"""Per-source normalization for RainPro-8 (Taiwan) inputs.

The paper applies min-max normalization to every source (Sec. A.2). Bounds
below are physically-reasonable defaults for the sources with a well-known
range (radar dBZ, IR brightness temperature). Any GFS variables
(`--data.include_gfs true`) span very different units/scales; no default
bounds are given for those -- they should be computed from training-set
statistics and supplied via `norm_bounds` before real training. Left
un-normalized (pass-through) until then, since a wrong guessed bound is worse
than none.
"""

from __future__ import annotations

import numpy as np

DBZ_RANGE = (-1.0, 64.0)  # paper clips radar reflectivity to this range
BRIGHTNESS_TEMP_RANGE = (180.0, 320.0)  # K, typical IR window range

DEFAULT_NORM_BOUNDS: dict[str, tuple[float, float]] = {
    "max_dbz": DBZ_RANGE,
    "dbz": DBZ_RANGE,
    **{band: BRIGHTNESS_TEMP_RANGE for band in
       ["B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16"]},
}


def minmax_normalize(
    x: np.ndarray,
    bounds: tuple[float, float] | None,
    clip: bool = True,
) -> np.ndarray:
    if bounds is None:
        return x
    vmin, vmax = bounds
    if clip:
        x = np.clip(x, vmin, vmax)
    return (x - vmin) / (vmax - vmin)
