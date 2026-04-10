from dataclasses import dataclass

import torch


@dataclass
class EvalOutputs:
    forecast: torch.Tensor
    target: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    loss: torch.Tensor | None = None


@dataclass(frozen=True)
class EvalRequest:
    only_loss: bool = False
    need_forecast: bool = False
    need_target: bool = False
    need_loss: bool = False
    need_probs: bool = False
