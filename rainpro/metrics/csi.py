import torch
import torchmetrics
from einops import rearrange
from torchmetrics.utilities.compute import _safe_divide

from rainpro.modules.utils import EvalOutputs


class CriticalSuccessIndex(torchmetrics.Metric):
    is_differentiable = False
    full_state_update = False
    higher_is_better = True

    hits: torch.Tensor
    false_guesses: torch.Tensor

    def __init__(
        self,
        num_lead_times: int,
        thresholds: list[int],
        average_time: bool = True,
    ):
        super().__init__()
        self.thresholds = thresholds
        self.average_time = average_time

        pad = len(str(int(max(self.thresholds))))
        self.padded_names = [str(int(t)).zfill(pad) for t in self.thresholds]

        zeros_shape = (len(thresholds), num_lead_times)
        self.add_state(
            "hits",
            default=torch.zeros(zeros_shape),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_guesses",
            default=torch.zeros(zeros_shape),
            dist_reduce_fx="sum",
        )

    def update(self, eval_outputs: EvalOutputs):
        preds = eval_outputs.forecast
        target = eval_outputs.target
        assert target is not None

        thresholds_tensor = torch.tensor(self.thresholds, device=preds.device).view(
            -1, 1, 1, 1, 1, 1
        )

        preds_bin = preds.unsqueeze(0) >= thresholds_tensor
        target_bin = target.unsqueeze(0) >= thresholds_tensor

        # Reshape to [threshold, B, T, pixels]
        preds_bin = rearrange(preds_bin, "th b t c h w -> th b t (c h w)")
        target_bin = rearrange(target_bin, "th b t c h w -> th b t (c h w)")

        # Compute metrics as [th, B, T]
        hits = (preds_bin & target_bin).sum(dim=-1)
        false_guesses = (preds_bin ^ target_bin).sum(-1)  # XOR

        # Update metrics as [th, T]
        self.hits += hits.sum(dim=1)
        self.false_guesses += false_guesses.sum(dim=1)

    def compute(self):
        return self._compute(reduce_mean=True)

    def _compute(self, reduce_mean=True):

        csi = _safe_divide(
            self.hits,
            self.hits + self.false_guesses,
        )

        # CSI: [th, T]
        if not reduce_mean or not self.average_time:
            return csi

        return csi.mean(dim=1).mean()
