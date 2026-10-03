"""Synthetic checks for `rainpro.baselines.optical_flow`: Gaussian echo blobs
translating at a known, sub-pixel velocity on the real `radar_4km` /
`target_2km` geometry. Run with `pytest tests/test_optical_flow.py`."""

import torch

from rainpro.baselines.optical_flow import OpticalFlowBaseline
from rainpro.data.normalize import DBZ_RANGE
from rainpro.data.rainpro8_sources import build_taiwan_sources

# (x, y) in 4 km input pixels per 10 min, i.e. ~13 m/s
VELOCITY = (1.6, -0.9)
BLOBS = [(100.0, 120.0, 45.0, 6.0), (150.0, 140.0, 35.0, 9.0), (120.0, 90.0, 50.0, 4.0)]


def _field(km_x: torch.Tensor, km_y: torch.Tensor, t_steps: float) -> torch.Tensor:
    """dBZ at physical positions (km from the canvas centre) after `t_steps`
    frame intervals; blob centres are given in input pixels."""
    out = torch.zeros_like(km_x)
    for cx, cy, peak, sigma in BLOBS:
        bx = (cx - 127.5 + VELOCITY[0] * t_steps) * 4.0
        by = (cy - 127.5 + VELOCITY[1] * t_steps) * 4.0
        out = torch.maximum(out, peak * torch.exp(-((km_x - bx) ** 2 + (km_y - by) ** 2) / (2 * (sigma * 4.0) ** 2)))
    return out


def _canvas_km(size_px: int, res_km: float) -> tuple[torch.Tensor, torch.Tensor]:
    km = (torch.arange(size_px, dtype=torch.float32) - (size_px - 1) / 2) * res_km
    ky, kx = torch.meshgrid(km, km, indexing="ij")
    return kx, ky


def _setup():
    sources = build_taiwan_sources(include_satellite=False)
    baseline = OpticalFlowBaseline.from_sources(sources)
    spec = sources["radar_4km"]
    kx, ky = _canvas_km(spec.size_px, spec.resolution_km)
    steps = [o / 10 for o in spec.offsets_min]  # -6 .. 0
    dbz = torch.stack([_field(kx, ky, s) for s in steps])[None, :, None]  # (1, 7, 1, H, W)
    lo, hi = DBZ_RANGE
    radar = (dbz.clamp(lo, hi) - lo) / (hi - lo)
    return sources, baseline, radar


def test_motion_recovers_translation():
    _, baseline, radar = _setup()
    lo, hi = DBZ_RANGE
    dbz = radar[:, :, 0] * (hi - lo) + lo
    flow = baseline.motion(dbz[:, -(baseline.n_pairs + 1) :])[0]  # (2, H, W)
    echo = dbz[0, -1] > 20
    for axis, v in enumerate(VELOCITY):
        assert abs(flow[axis][echo].mean().item() - v) < 0.15


def test_forecast_tracks_moving_echo():
    sources, baseline, radar = _setup()
    forecast = baseline({"radar_4km": radar})
    target = sources["target_2km"]
    assert forecast.shape == (1, target.timesteps, 1, target.size_px, target.size_px)

    kx, ky = _canvas_km(target.size_px, target.resolution_km)
    for li, lead in enumerate(target.offsets_min[:12]):  # first 2 h
        truth = _field(kx, ky, lead / 10)
        pred = forecast[0, li, 0]
        hits = ((pred >= 30) & (truth >= 30)).sum()
        misses_fa = ((pred >= 30) ^ (truth >= 30)).sum()
        csi = (hits / (hits + misses_fa)).item()
        assert csi > 0.85, f"lead {lead}: CSI@30 {csi:.3f}"

    # Guard against a test persistence could pass: by +2 h the echo has moved
    # well off its t0 footprint.
    truth = _field(kx, ky, 12)
    t0 = _field(kx, ky, 0)
    hits = ((t0 >= 30) & (truth >= 30)).sum()
    persistence_csi = (hits / (hits + ((t0 >= 30) ^ (truth >= 30)).sum())).item()
    assert persistence_csi < 0.5


def test_static_and_empty_input():
    sources, baseline, _ = _setup()
    spec = sources["radar_4km"]
    # all fill (out of coverage / missing frames) -> no echo anywhere
    empty = torch.zeros(2, spec.timesteps, 1, spec.size_px, spec.size_px)
    out = baseline({"radar_4km": empty})
    assert torch.isfinite(out).all() and out.abs().max() == 0
