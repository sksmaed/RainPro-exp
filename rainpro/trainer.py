import os
from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from rainpro.callbacks.factory import create_callbacks


class RainProTrainer(pl.Trainer):
    def __init__(
        self,
        run_name: str,
        devices: list[int] | str | int = "auto",
        precision: Literal["32", "bf16-mixed"] = "32",
        gradient_clip_val: float = 0.0,
        max_epochs: int = 50,
        log_every_n_steps: int | None = 50,
        early_stopping_patience: int | None = None,
        root_dir: str = "runs",
        project: str = "rainpro",
        num_animations: int | None = 20,
        callbacks: list[pl.Callback] | pl.Callback | None = None,
        **kwargs,
    ):
        _logger = WandbLogger(name=run_name, project=project)

        callbacks = create_callbacks(
            os.path.join(root_dir, run_name, "checkpoints"),
            early_stopping_patience,
            num_animations,
        )

        super().__init__(
            default_root_dir=root_dir,
            devices=devices,
            callbacks=callbacks,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            num_sanity_val_steps=0,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            logger=_logger,
            **kwargs,
        )
