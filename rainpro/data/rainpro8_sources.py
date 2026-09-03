"""RainPro-8 (Taiwan) data source specification: observation-only baseline
(QPESUMS radar + STA_H8 satellite), with GFS kept available as an optional
16 km-tier input behind the `include_gfs` flag.

Maps `docs/rainpro_dataset.md` onto the input layout expected by
`rainpro.network.rainpro8.RainPro` (see `rainpro8.ipynb` for the reference
European layout this mirrors):

| role (RainPro-8 tier) | Taiwan source            | resolution | notes |
| ---------------------- | ------------------------- | ---------- | ----- |
| target_2km              | QPESUMS max dBZ           | 2 km       | Marshall-Palmer -> mm/h |
| radar_4km                | QPESUMS (downsampled)     | 4 km       | -60..0 min |
| radar_8km                | QPESUMS (downsampled)     | 8 km       | t=0 |
| satellite_8km             | STA_H8 (9 IR bands)       | 8 km       | -120..-60 min @ 10 min |
| "16km" tier (NWP, optional) | GFS (`include_gfs=True`) | -> 16 km   | see below |
|                              | GFS forecast (`include_gfs=True`)| -> 16 km | +60..+480 min |

This is deliberately the smallest obs-only baseline (QPESUMS + STA_H8) plus a
single on/off switch for GFS, to compare the two arms directly before adding
any other source. When `include_gfs=False`, there is no 16 km tier at all;
`rainpro.network.rainpro8.RainPro` (via `in_dims`, see `tier_channels` below)
handles a 0-channel/absent 16 km tier by skipping that branch of the encoder.

Spatial canvas sizes (target 512 km, radar_4km context 1024 km, 8km/16km-tier
context 1536 km) intentionally reuse the exact geometry of the original
European RainPro-8 (see `rainpro8.ipynb`), since `rainpro.network.rainpro8.RainPro`
hard-assumes those integer relationships between tiers (each tier is exactly
2x downsampled from the previous one). Taiwan's native source coverage is
smaller/non-square (e.g. QPESUMS ~565x780 km) than these canvases; sources
are regridded (`rainpro.data.regrid`) onto the common square canvases and
pixels outside native coverage are filled as missing (NaN for the target,
`fill_value` for inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

Tier = str  # "target_2km" | "4km" | "8km" | "16km"

STA_H8_BANDS = [f"B{i:02d}" for i in range(8, 17)]  # B08..B16, 9 IR bands

# Paper App. I (Table 11): the 122 `gfs_16km` channels kept from the full GFS
# field set, flattened to one canonical name per (variable, level) pair --
# e.g. "TMP_850mb", "TMP_surface" -- with bare GRIB2 codes (e.g. "PRATE") for
# variables that only have a single level in Table 11. These are canonical
# names, not necessarily the raw GFS zarr store's own field names; map real
# names onto them via `variable_aliases` (see `RainPro8Dataset`) if the store
# uses different conventions (e.g. no per-level suffix, different level units).
GFS_ANALYSIS_VARIABLES: tuple[str, ...] = (
    "4LFTX",
    *(f"ABSV_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb")),
    *(f"CAPE_{lvl}" for lvl in ("surface", "180-0mb", "90-0mb", "255-0mb")),
    "CFRZR",
    "CICEP",
    *(f"CIN_{lvl}" for lvl in ("surface", "180-0mb", "90-0mb", "255-0mb")),
    *(f"CLMR_{lvl}" for lvl in ("250mb", "500mb", "850mb", "1000mb")),
    "CPOFP",
    "CRAIN",
    "CSNOW",
    "CWAT",
    "DPT",
    *(f"DZDT_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb")),
    *(f"GRLE_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb")),
    "GUST",
    *(f"HGT_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb", "surface", "trop")),
    "HPBL",
    "ICEG",
    "ICETK",
    *(f"ICMR_{lvl}" for lvl in ("250mb", "500mb", "850mb", "1000mb")),
    "LCDC",
    "LFTX",
    "MCDC",
    "MSLET",
    "PLPL",
    "PRATE",
    *(f"PRES_{lvl}" for lvl in ("surface", "trop")),
    "PRMSL",
    "PWAT",
    "REFC",
    *(f"RH_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb", "2m", "atm")),
    *(f"RWMR_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb")),
    *(f"SNMR_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb")),
    "SNOD",
    *(f"SPFH_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb", "2m")),
    *(f"TCDC_{lvl}" for lvl in ("250mb", "500mb", "850mb", "1000mb", "atm")),
    *(f"TMP_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb", "surface", "2m", "trop")),
    *(f"UGRD_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb", "10m", "trop")),
    "USTM",
    *(f"VGRD_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb", "10m", "trop")),
    "VIS",
    "VSTM",
    *(f"VVEL_{lvl}" for lvl in ("100mb", "250mb", "500mb", "850mb", "1000mb")),
    "VWSH",
    "WEASD",
)
assert len(GFS_ANALYSIS_VARIABLES) == 122


@dataclass(frozen=True)
class SourceSpec:
    name: str
    tier: Tier
    resolution_km: float
    size_km: float
    offsets_min: Sequence[int]
    variables: Sequence[str] = ()  # single-channel (2D, or already-flat) variables
    variables_3d: Sequence[str] = ()  # level-resolved variables, see `levels`
    levels: Sequence[int] = ()  # pressure levels shared by all `variables_3d`
    fill_value: float = 0.0

    @property
    def size_px(self) -> int:
        return round(self.size_km / self.resolution_km)

    @property
    def timesteps(self) -> int:
        return len(self.offsets_min)

    @property
    def channels(self) -> int:
        return len(self.variables) + len(self.variables_3d) * len(self.levels)


def build_taiwan_sources(
    include_gfs: bool = False,
    gfs_variables: Sequence[str] = GFS_ANALYSIS_VARIABLES,
    gfs_forecast_variables: Sequence[str] = ("PRATE",),
) -> dict[str, SourceSpec]:
    """Build the Taiwan RainPro-8 source spec dict, keyed like the model's `sources`.

    `include_gfs` toggles the only NWP input, GFS (analysis + forecast), which
    fills the 16 km tier. When `False` (the obs-only arm), there is no 16 km
    tier at all -- only QPESUMS (target/radar) and STA_H8 (satellite) feed the
    model.

    `gfs_variables` defaults to `GFS_ANALYSIS_VARIABLES`, the paper's 122-channel
    App. I selection (see that constant's docstring re: `variable_aliases` if
    the real GFS store names fields differently). Pass a different sequence
    (or `()`) to override -- `gfs_variables`/`gfs_forecast_variables` stay
    configurable so the exact channel count can be tuned without code changes.

    `satellite_8km` samples every 10 min (7 steps) rather than the paper's
    15 min/5-step EUMETSAT cadence: STA_H8 is a native 10-minute product,
    unlike EUMETSAT's ~1h-delayed feed that motivated the paper's coarser,
    -60min-starting window (Sec. A.2).
    """
    sources: dict[str, SourceSpec] = {
        "target_2km": SourceSpec(
            name="target_2km",
            tier="target_2km",
            resolution_km=2,
            size_km=512,
            offsets_min=tuple(range(10, 370, 10)),  # 0-6h @ 10 min, 36 steps
            variables=("max_dbz",),
        ),
        "radar_4km": SourceSpec(
            name="radar_4km",
            tier="4km",
            resolution_km=4,
            size_km=1024,
            offsets_min=tuple(range(-60, 10, 10)),  # -60..0 min, 7 steps
            variables=("max_dbz",),
        ),
        "radar_8km": SourceSpec(
            name="radar_8km",
            tier="8km",
            resolution_km=8,
            size_km=1536,
            offsets_min=(0,),
            variables=("max_dbz",),
        ),
        "satellite_8km": SourceSpec(
            name="satellite_8km",
            tier="8km",
            resolution_km=8,
            size_km=1536,
            offsets_min=tuple(range(-120, -50, 10)),  # -120..-60 min @ 10 min, 7 steps
            variables=tuple(STA_H8_BANDS),
        ),
    }

    if include_gfs:
        if not gfs_variables:
            import warnings

            warnings.warn(
                "include_gfs=True but gfs_variables is empty -- gfs_16km will have 0 "
                "channels. Pass `gfs_variables` (defaults to `GFS_ANALYSIS_VARIABLES`, "
                "see paper App. I) to actually use it.",
                stacklevel=2,
            )
        sources["gfs_16km"] = SourceSpec(
            name="gfs_16km",
            tier="16km",
            resolution_km=16,
            size_km=1536,
            offsets_min=(0,),
            variables=tuple(gfs_variables),
        )
        sources["gfs_forecast_16km"] = SourceSpec(
            name="gfs_forecast_16km",
            tier="16km",
            resolution_km=16,
            size_km=1536,
            offsets_min=tuple(range(60, 540, 60)),  # +60..+480 min, 8 steps
            variables=tuple(gfs_forecast_variables),
        )

    return sources


def tier_channels(sources: dict[str, SourceSpec], tier: str) -> int:
    return sum(s.timesteps * s.channels for s in sources.values() if s.tier == tier)


def context_sizes(sources: dict[str, SourceSpec]) -> tuple[int, int]:
    """Context (padding, per side) for the 4km and 8km tiers wrt. the 2km target,
    following the same formula as `rainpro8.ipynb`."""
    target_px = sources["target_2km"].size_px
    px_4km = next(s.size_px for s in sources.values() if s.tier == "4km")
    px_8km = next(s.size_px for s in sources.values() if s.tier == "8km")
    context_size_4km = (px_4km - target_px // 2) // 2
    context_size_8km = (px_8km - target_px // 4) // 2
    return context_size_4km, context_size_8km
