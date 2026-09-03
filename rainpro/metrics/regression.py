"""Per-lead-time MAE/MSE, masked at NaN target (QPESUMS out-of-coverage).

Kept separate from `torchmetrics.MeanAbsoluteError`/`MeanSquaredError`
because those reduce to a single global scalar; the per-lead-time breakdown
needed here follows the same pooled-accumulation pattern as
`rainpro.metrics.csi.CriticalSuccessIndex` (sum of errors / count, not a
mean of per-batch means).
"""

import torch
import torchmetrics

from rainpro.modules.utils import EvalOutputs


class LeadTimeMAEMSE(torchmetrics.Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = False

    sum_abs_err: torch.Tensor
    sum_sq_err: torch.Tensor
    count: torch.Tensor

    def __init__(self, num_lead_times: int):
        super().__init__()
        self.add_state("sum_abs_err", default=torch.zeros(num_lead_times), dist_reduce_fx="sum")
        self.add_state("sum_sq_err", default=torch.zeros(num_lead_times), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(num_lead_times), dist_reduce_fx="sum")

    def update(self, eval_outputs: EvalOutputs):
        preds = eval_outputs.forecast  # (B, T, C, H, W)
        target = eval_outputs.target
        assert target is not None

        valid = ~torch.isnan(target)
        err = torch.where(valid, preds - target, torch.zeros_like(target))

        self.sum_abs_err += err.abs().sum(dim=(0, 2, 3, 4))
        self.sum_sq_err += (err**2).sum(dim=(0, 2, 3, 4))
        self.count += valid.float().sum(dim=(0, 2, 3, 4))

    def compute(self) -> torch.Tensor:
        mae = self.sum_abs_err / self.count.clamp(min=1)
        return mae.mean()

    def full(self) -> dict[str, torch.Tensor]:
        count = self.count.clamp(min=1)
        return {
            "MAE": self.sum_abs_err / count,
            "MSE": self.sum_sq_err / count,
        }
