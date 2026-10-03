from dataclasses import dataclass

import torch


@dataclass
class EvalOutputs:
    forecast: torch.Tensor
    target: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    # Reference forecast on the same grid as `forecast`, for side-by-side
    # visualization only (test: the optical-flow baseline; see
    # `RainPro8Module.test_step` and `rainpro.callbacks.log_animations`).
    baseline: torch.Tensor | None = None


@dataclass(frozen=True)
class EvalRequest:
    only_loss: bool = False
    need_forecast: bool = False
    need_target: bool = False
    need_loss: bool = False
    need_probs: bool = False
