"""Optical-flow extrapolation baseline (the classical "P0/P1" nowcast of
`docs/rainpro_tw_evaluation.md` Stage 4.1), scored in the test loop next to
the model.

Same recipe as PySTEPS' default extrapolation nowcast (the baseline the
paper compares against): estimate one motion field from the most recent radar
frames, hold it constant, and advect the t0 frame along it with semi-
Lagrangian backward trajectories. Written in plain torch rather than calling
pysteps so it runs batched on the GPU inside `test_step` with no extra
dependencies (pysteps is numpy/OpenCV, one sample at a time on the CPU).

Motion: dense pyramidal Lucas-Kanade between consecutive `radar_4km` frames
(10 min apart), the last `n_pairs` pairs pooled. LK only measures motion where
there is texture, i.e. inside echo; everywhere else the field is filled by
confidence-weighted Gaussian smoothing (normalized convolution), falling back
to the frame's mean motion far from any echo. Without that fill, backward
advection would sample zero motion just ahead of a moving echo and stall its
leading edge.

Input is the model's own `radar_4km` source, so the baseline sees no more
radar than the model does: 4 km resolution (vs. the 2 km target), but the
whole 1024 km canvas, so echo up to 256 km outside the target domain can
advect into it. Trajectories are traced directly from the 2 km target pixel
centres, which avoids a blocky 4 km -> 2 km upsample. Each lead time samples
the t0 frame once (bilinear) at the end of its trajectory, so intensity is
not re-smoothed step after step.

`radar_4km` arrives min-max normalized with out-of-coverage pixels at
`fill_value` 0 (= -1 dBZ); both that and QPESUMS' no-echo 0 dBZ come out as
0 dBZ here, the same "no echo" convention as `scripts/eval_persistence.py`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rainpro.data.normalize import DEFAULT_NORM_BOUNDS
from rainpro.data.rainpro8_sources import SourceSpec


class OpticalFlowBaseline:
    def __init__(
        self,
        input_spec: SourceSpec,
        target_spec: SourceSpec,
        dbz_bounds: tuple[float, float],
        n_pairs: int = 3,
        levels: int = 3,
        iterations: int = 4,
        presmooth_sigma: float = 1.0,
        window_sigma: float = 3.0,
        fill_sigma: float = 8.0,
        min_dbz: float = 10.0,
        regularization: float = 1.0,
    ):
        """`n_pairs`: consecutive frame pairs pooled into one motion field.
        `levels`/`iterations`: LK pyramid depth (each level halves the grid;
        3 levels at 4 km follow ~10 px = 40 km per 10 min) and warp-and-solve
        rounds per level. Sigmas are in input pixels: `window_sigma` is the LK
        integration window, `fill_sigma` the motion-fill smoothing (8 px =
        32 km). `min_dbz`: reflectivity is floored here before estimating
        motion, so clear-air noise and the 0 / -1 dBZ fill carry no texture.
        `regularization` (dBZ/px)^2 is added to the LK normal equations so
        flat regions get no update instead of an ill-posed one."""
        offsets = list(input_spec.offsets_min)
        dt = offsets[1] - offsets[0]
        assert offsets[-1] == 0 and all(b - a == dt for a, b in zip(offsets, offsets[1:])), (
            f"{input_spec.name} offsets must be evenly spaced and end at t0, got {offsets}"
        )
        assert len(offsets) > n_pairs, f"{input_spec.name} has {len(offsets)} frames, need {n_pairs + 1}"
        leads = list(target_spec.offsets_min)
        assert all(lead % dt == 0 for lead in leads) and leads == sorted(leads), (
            f"target leads {leads} must be increasing multiples of {dt} min"
        )

        self.input_name = input_spec.name
        self.input_size = (input_spec.size_px, input_spec.size_px)
        self.lead_steps = [lead // dt for lead in leads]
        self.dbz_bounds = dbz_bounds
        self.n_pairs = n_pairs
        self.levels = levels
        self.iterations = iterations
        self.presmooth_sigma = presmooth_sigma
        self.window_sigma = window_sigma
        self.fill_sigma = fill_sigma
        self.min_dbz = min_dbz
        self.regularization = regularization
        self.target_grid = _nested_grid(target_spec, input_spec)

    @classmethod
    def from_sources(
        cls,
        sources: dict[str, SourceSpec],
        norm_bounds: dict[str, tuple[float, float]] | None = None,
        input_name: str = "radar_4km",
        **kwargs,
    ) -> "OpticalFlowBaseline":
        input_spec = sources[input_name]
        # Same lookup RainPro8Dataset normalizes with: defaults, then overrides.
        bounds = {**DEFAULT_NORM_BOUNDS, **(norm_bounds or {})}[input_spec.variables[0]]
        return cls(input_spec, sources["target_2km"], tuple(bounds), **kwargs)

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forecast in raw dBZ, (B, T_out, 1, H, W) on the target grid."""
        lo, hi = self.dbz_bounds
        dbz = (batch[self.input_name][:, :, 0].float() * (hi - lo) + lo).clamp(min=0.0)
        flow = self.motion(dbz[:, -(self.n_pairs + 1) :])
        return self.extrapolate(dbz[:, -1], flow)

    def motion(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B, K+1, H, W) dBZ, oldest first -> flow (B, 2, H, W), (x, y)
        in input pixels per frame interval."""
        b, k1, h, w = frames.shape
        x = frames.clamp(min=self.min_dbz) - self.min_dbz
        x = _gaussian_blur(x.reshape(b * k1, 1, h, w), self.presmooth_sigma).view(b, k1, h, w)
        i0 = x[:, :-1].reshape(-1, 1, h, w)
        i1 = x[:, 1:].reshape(-1, 1, h, w)

        flow, conf = self._pyramidal_lk(i0, i1)
        flow = flow.view(b, k1 - 1, 2, h, w)
        conf = conf.view(b, k1 - 1, 1, h, w)

        num = _gaussian_blur((conf * flow).sum(dim=1), self.fill_sigma)
        den = _gaussian_blur(conf.sum(dim=1), self.fill_sigma)
        mean_flow = num.sum(dim=(2, 3), keepdim=True) / den.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        # Where nearby confidence is small relative to the frame's best, lean
        # on the frame-mean motion instead of a noisy local estimate.
        eps = 0.01 * den.amax(dim=(2, 3), keepdim=True) + 1e-6
        return (num + eps * mean_flow) / (den + eps)

    def extrapolate(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """frame (B, H, W) dBZ at t0, flow (B, 2, H, W) px/step on the input
        grid -> (B, T_out, 1, Ht, Wt) on the target grid."""
        b = frame.shape[0]
        h, w = self.input_size
        to_norm = torch.tensor([2.0 / w, 2.0 / h], device=flow.device).view(1, 2, 1, 1)
        flow = flow * to_norm
        coords = self.target_grid.to(frame.device).expand(b, -1, -1, -1)

        out, step = [], 0
        for lead_step in self.lead_steps:
            while step < lead_step:
                v = F.grid_sample(flow, coords, mode="bilinear", padding_mode="border", align_corners=False)
                coords = coords - v.permute(0, 2, 3, 1)
                step += 1
            # Trajectories leaving the canvas read 0 dBZ (no echo).
            out.append(
                F.grid_sample(frame[:, None], coords, mode="bilinear", padding_mode="zeros", align_corners=False)
            )
        return torch.stack(out, dim=1)

    def _pyramidal_lk(self, i0: torch.Tensor, i1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pyr0, pyr1 = [i0], [i1]
        for _ in range(self.levels - 1):
            pyr0.append(F.avg_pool2d(pyr0[-1], 2))
            pyr1.append(F.avg_pool2d(pyr1[-1], 2))

        flow = torch.zeros_like(pyr0[-1]).expand(-1, 2, -1, -1).contiguous()
        for a, b in zip(reversed(pyr0), reversed(pyr1)):
            if flow.shape[-2:] != a.shape[-2:]:
                scale = torch.tensor(
                    [a.shape[-1] / flow.shape[-1], a.shape[-2] / flow.shape[-2]], device=flow.device
                ).view(1, 2, 1, 1)
                flow = F.interpolate(flow, size=a.shape[-2:], mode="bilinear", align_corners=False) * scale
            for _ in range(self.iterations):
                flow, conf = self._lk_step(a, b, flow)
        return flow, conf

    def _lk_step(
        self, i0: torch.Tensor, i1: torch.Tensor, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One warp-and-solve round: linearize I1(x + flow + d) = I0(x) and
        solve the windowed 2x2 normal equations for d."""
        i1w = _warp(i1, flow)
        ix, iy = _gradients(0.5 * (i0 + i1w))
        it = i1w - i0
        sxx, sxy, syy, sxt, syt = _gaussian_blur(
            torch.cat([ix * ix, ix * iy, iy * iy, ix * it, iy * it], dim=1), self.window_sigma
        ).unbind(dim=1)

        # Smaller eigenvalue-like texture measure (det / trace): high only at
        # corners/blobs, ~0 along straight edges (aperture problem) and in
        # flat regions.
        conf = ((sxx * syy - sxy**2) / (sxx + syy + 1e-6)).clamp(min=0.0)[:, None]

        sxx = sxx + self.regularization
        syy = syy + self.regularization
        det = sxx * syy - sxy**2
        du = (sxy * syt - syy * sxt) / det
        dv = (sxy * sxt - sxx * syt) / det
        return flow + torch.stack([du, dv], dim=1), conf


def _nested_grid(inner: SourceSpec, outer: SourceSpec) -> torch.Tensor:
    """`inner`'s pixel centres in `outer`'s `grid_sample` coordinates
    (align_corners=False), (1, H_in, W_in, 2) as (x, y). Both canvases are
    centred on the same point (`rainpro.data.regrid.target_grid`), with rows
    along latitude and columns along longitude."""
    n_in, n_out = inner.size_px, outer.size_px
    km = (torch.arange(n_in, dtype=torch.float32) - (n_in - 1) / 2) * inner.resolution_km
    px = km / outer.resolution_km + (n_out - 1) / 2
    norm = (2 * px + 1) / n_out - 1
    gy, gx = torch.meshgrid(norm, norm, indexing="ij")
    return torch.stack([gx, gy], dim=-1)[None]


def _warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample `img` at x + flow (flow in pixels, (x, y))."""
    _, _, h, w = img.shape
    ys = (2 * torch.arange(h, device=img.device, dtype=img.dtype) + 1) / h - 1
    xs = (2 * torch.arange(w, device=img.device, dtype=img.dtype) + 1) / w - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx + flow[:, 0] * (2.0 / w), gy + flow[:, 1] * (2.0 / h)], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    px = F.pad(x, (1, 1, 0, 0), mode="replicate")
    py = F.pad(x, (0, 0, 1, 1), mode="replicate")
    return 0.5 * (px[..., 2:] - px[..., :-2]), 0.5 * (py[..., 2:, :] - py[..., :-2, :])


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian, per channel, replicate-padded."""
    radius = max(1, int(3 * sigma + 0.5))
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k**2) / (2 * sigma**2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x
