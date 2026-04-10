import lightning as L
import lightning.pytorch as pl
import matplotlib as mpl
import numpy as np
import torch
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

import wandb
from rainpro.modules.utils import EvalOutputs


class LogAnimations(L.Callback):
    def __init__(self, num_animations: int):
        super().__init__()
        self.num_animations = num_animations
        self.cmap, self.norm = _vil_cmap()
        self.min_interesting_value = 160
        self.min_interesting_count = 10
        self.animations: list[Tensor] = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self._add_animations(outputs)

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self._add_animations(outputs)

    def on_validation_end(self, trainer, pl_module):
        self._log_animations("val", trainer)

    def on_test_end(self, trainer, pl_module):
        self._log_animations("test", trainer)

    @rank_zero_only
    def _add_animations(self, outputs: EvalOutputs):
        if len(self.animations) >= self.num_animations:
            return

        preds = outputs.forecast.cpu()
        targets = outputs.target
        assert targets is not None
        targets = targets.cpu()

        for pred, target in zip(preds, targets):
            if len(self.animations) >= self.num_animations:
                break

            if (target >= self.min_interesting_value).sum() < self.min_interesting_count:
                continue

            pred = self._colorize(pred)
            target = self._colorize(target)

            # concatenate side-by-side (W dimension)
            animation = torch.cat([pred, target], dim=2)

            # upsample for visibility
            animation = torch.nn.functional.interpolate(
                animation.permute(0, 3, 1, 2),
                scale_factor=5,
                mode="nearest",
            ).permute(0, 2, 3, 1)

            self.animations.append(animation)

    @rank_zero_only
    def _log_animations(self, stage: str, trainer: pl.Trainer):
        if not self.animations:
            return

        logger = trainer.logger
        if not isinstance(logger, WandbLogger):
            self.animations.clear()
            return

        log_dict = {"global_step": trainer.global_step}

        for idx, animation in enumerate(self.animations):
            # animation: (T, H, W, C) tensor, uint8
            animation_np = rearrange(animation.cpu().numpy(), "t h w c -> 1 t c h w")  # shape: (1, T, C, H, W)
            log_dict[f"{stage}_animations/{idx}"] = wandb.Video(
                animation_np,
                fps=8,
                format="mp4"
            )

        logger.experiment.log(log_dict)
        self.animations.clear()

    def _colorize(self, rain_rates: Tensor) -> Tensor:
        """Convert (T, 1, H, W) → (T, H, W, 3) uint8"""
        rain_np = rain_rates.squeeze(1).cpu().numpy()
        rgba = self.cmap(self.norm(rain_np))

        alpha = rgba[..., 3:4]
        rgb = rgba[..., :3] * alpha

        return torch.from_numpy((rgb * 255).astype(np.uint8))


def _vil_cmap():
    color_map = [
        [0, 0, 0],
        [0.30196, 0.30196, 0.30196],
        [0.15686, 0.74510, 0.15686],
        [0.09804, 0.58824, 0.09804],
        [0.03922, 0.41176, 0.03922],
        [0.03922, 0.29412, 0.03922],
        [0.96078, 0.96078, 0.0],
        [0.92941, 0.67451, 0.0],
        [0.94118, 0.43137, 0.0],
        [0.62745, 0.0, 0.0],
        [0.90588, 0.0, 1.0],
    ]

    bounds = [
        0.0,
        16.0,
        31.0,
        59.0,
        74.0,
        100.0,
        133.0,
        160.0,
        181.0,
        219.0,
        255.0,
    ]

    cmap = mpl.colors.ListedColormap(color_map)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm
