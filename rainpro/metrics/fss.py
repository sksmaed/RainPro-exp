"""Fractions Skill Score (FSS), pooled per (threshold, window, lead_time).

Neighborhood window is a square side length in {1, 2, 4, 8} grid cells
(window=1 has no pooling and should numerically match
`rainpro.metrics.csi.CriticalSuccessIndex` restricted to the same threshold
-- a useful correctness check, see `tests/test_metrics.py`).

NaN handling (`target` carries NaN outside QPESUMS coverage, see
`rainpro.data.rainpro8_dataset`): the *center* pixel's validity gates whether
that location contributes to the pooled sums at all (same convention as
`rainpro.metrics.contingency.ContingencyMetrics`). The neighborhood-fraction
pooling itself does not separately exclude invalid neighbors -- out-of-
coverage neighbors are treated as unconditionally 0 (no rain) inside the
pooling window, the same approximation `avg_pool2d`'s zero-padding already
makes at the domain edges.
"""

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.utilities.compute import _safe_divide

from rainpro.modules.utils import EvalOutputs

DEFAULT_WINDOWS = (1, 2, 4, 8)


class FractionsSkillScore(torchmetrics.Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = True

    sum_sq_diff: torch.Tensor
    sum_sq_ref: torch.Tensor

    def __init__(
        self,
        num_lead_times: int,
        thresholds: list[float],
        windows: tuple[int, ...] = DEFAULT_WINDOWS,
    ):
        super().__init__()
        self.thresholds = thresholds
        self.windows = list(windows)

        pad = len(str(int(max(thresholds))))
        self.labels = [str(int(t)).zfill(pad) for t in thresholds]

        zeros_shape = (len(thresholds), len(self.windows), num_lead_times)
        self.add_state("sum_sq_diff", default=torch.zeros(zeros_shape), dist_reduce_fx="sum")
        self.add_state("sum_sq_ref", default=torch.zeros(zeros_shape), dist_reduce_fx="sum")

    def update(self, eval_outputs: EvalOutputs):
        preds = eval_outputs.forecast  # (B, T, C, H, W)
        target = eval_outputs.target
        assert target is not None

        b, t, c, h, w = preds.shape
        valid = ~torch.isnan(target)
        # A value that never exceeds any (positive) threshold, so NaN pixels
        # binarize to 0 before pooling; `valid` still gates the final sums.
        target_safe = torch.nan_to_num(target, nan=-float("inf"))

        preds_flat = preds.reshape(b * t, c, h, w)
        target_flat = target_safe.reshape(b * t, c, h, w)
        valid_flat = valid.reshape(b * t, c, h, w)

        for wi, window in enumerate(self.windows):
            pad = window // 2
            for ti, threshold in enumerate(self.thresholds):
                fcst_bin = (preds_flat >= threshold).float()
                obs_bin = (target_flat >= threshold).float()

                fcst_frac = F.avg_pool2d(fcst_bin, kernel_size=window, stride=1, padding=pad)
                obs_frac = F.avg_pool2d(obs_bin, kernel_size=window, stride=1, padding=pad)
                # Even kernel sizes make avg_pool2d return one extra row/col;
                # crop back to (h, w) from the top-left.
                fcst_frac = fcst_frac[..., :h, :w]
                obs_frac = obs_frac[..., :h, :w]

                sq_diff = (fcst_frac - obs_frac) ** 2
                sq_ref = fcst_frac**2 + obs_frac**2

                zeros = torch.zeros_like(sq_diff)
                sq_diff = torch.where(valid_flat, sq_diff, zeros)
                sq_ref = torch.where(valid_flat, sq_ref, zeros)

                self.sum_sq_diff[ti, wi] += sq_diff.reshape(b, t, -1).sum(dim=(0, 2))
                self.sum_sq_ref[ti, wi] += sq_ref.reshape(b, t, -1).sum(dim=(0, 2))

    def compute(self) -> torch.Tensor:
        fss = 1 - _safe_divide(self.sum_sq_diff, self.sum_sq_ref)
        return fss.mean()

    def full(self) -> dict[str, torch.Tensor]:
        fss = 1 - _safe_divide(self.sum_sq_diff, self.sum_sq_ref)  # [threshold, window, T]
        return {f"FSS_w{window}": fss[:, wi] for wi, window in enumerate(self.windows)}
