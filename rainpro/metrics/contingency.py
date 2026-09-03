"""Pooled per-(threshold, lead_time) contingency-table metric: FBI/POD/FAR.

Extends `rainpro.metrics.csi.CriticalSuccessIndex`'s pooled accumulation
pattern, but keeps hits/misses/false_alarms/correct_negatives as four
separate buffers -- CSI's `false_guesses` combines misses+false_alarms via
XOR, which is enough for CSI but not enough to separate POD from FAR.

Unlike `CriticalSuccessIndex`, this masks out pixels where `target` is NaN
(QPESUMS out-of-coverage, see `rainpro.data.rainpro8_dataset`, `keep_nan=True`
for `target_2km`) -- `CriticalSuccessIndex` does not do this today (`NaN >=
threshold` evaluates `False`, silently counted as "observed no-rain"). This
is an intentional, documented divergence, not a fix retroactively applied to
CSI's already-reported numbers.
"""

import torch
import torchmetrics
from einops import rearrange
from torchmetrics.utilities.compute import _safe_divide

from rainpro.modules.utils import EvalOutputs


class ContingencyMetrics(torchmetrics.Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = None  # FBI is best ~1, POD/FAR have different optima

    hits: torch.Tensor
    misses: torch.Tensor
    false_alarms: torch.Tensor
    correct_negatives: torch.Tensor

    def __init__(self, num_lead_times: int, thresholds: list[float]):
        super().__init__()
        self.thresholds = thresholds

        pad = len(str(int(max(thresholds))))
        self.labels = [str(int(t)).zfill(pad) for t in thresholds]

        zeros_shape = (len(thresholds), num_lead_times)
        for name in ("hits", "misses", "false_alarms", "correct_negatives"):
            self.add_state(name, default=torch.zeros(zeros_shape), dist_reduce_fx="sum")

    def update(self, eval_outputs: EvalOutputs):
        preds = eval_outputs.forecast
        target = eval_outputs.target
        assert target is not None

        thresholds_tensor = torch.tensor(self.thresholds, device=preds.device).view(
            -1, 1, 1, 1, 1, 1
        )

        preds_bin = preds.unsqueeze(0) >= thresholds_tensor
        target_bin = target.unsqueeze(0) >= thresholds_tensor
        # NaN target: `NaN >= x` is False in target_bin, but `valid` zeroes
        # the pixel out of every bucket below regardless of that value.
        valid = (~torch.isnan(target)).unsqueeze(0)

        preds_bin = rearrange(preds_bin, "th b t c h w -> th b t (c h w)")
        target_bin = rearrange(target_bin, "th b t c h w -> th b t (c h w)")
        valid = rearrange(valid, "th b t c h w -> th b t (c h w)")

        hits = (preds_bin & target_bin & valid).sum(dim=-1)
        misses = (~preds_bin & target_bin & valid).sum(dim=-1)
        false_alarms = (preds_bin & ~target_bin & valid).sum(dim=-1)
        correct_negatives = (~preds_bin & ~target_bin & valid).sum(dim=-1)

        self.hits += hits.sum(dim=1)
        self.misses += misses.sum(dim=1)
        self.false_alarms += false_alarms.sum(dim=1)
        self.correct_negatives += correct_negatives.sum(dim=1)

    def compute(self) -> torch.Tensor:
        # Single scalar for Lightning `self.log_dict`; FBI/FAR available via `full()`.
        pod = _safe_divide(self.hits, self.hits + self.misses)
        return pod.mean()

    def full(self) -> dict[str, torch.Tensor]:
        fbi = _safe_divide(self.hits + self.false_alarms, self.hits + self.misses)
        pod = _safe_divide(self.hits, self.hits + self.misses)
        far = _safe_divide(self.false_alarms, self.hits + self.false_alarms)
        return {"FBI": fbi, "POD": pod, "FAR": far}
