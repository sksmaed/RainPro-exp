import os
from typing import Literal

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment

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
        animation_bounds: list[float] | None = None,
        callbacks: list[pl.Callback] | pl.Callback | None = None,
        auto_requeue: bool = False,
        checkpoint_interval_minutes: float | None = 30,
        **kwargs,
    ):
        _logger = WandbLogger(name=run_name, project=project)

        callbacks = create_callbacks(
            os.path.join(root_dir, run_name, "checkpoints"),
            early_stopping_patience,
            num_animations,
            animation_bounds,
            checkpoint_interval_minutes,
        )

        # When launched under `srun` inside an sbatch script, Lightning
        # auto-detects SLURMEnvironment() with its own default
        # (auto_requeue=True) unless a `ClusterEnvironment` is passed via
        # `plugins` explicitly -- that default is what prints "SLURM
        # auto-requeueing enabled. Setting signal handlers." in the job's
        # .err log and installs SIGUSR1/SIGTERM handlers that checkpoint and
        # `scontrol requeue` the job automatically near the time limit.
        # `auto_requeue=False` here opts out (a plain SLURM job timeout/kill
        # with no automatic resubmission); pass `auto_requeue=True` (or your
        # own `plugins=`) to restore Lightning's default behavior. Only
        # constructed when actually running under Slurm (`SLURMEnvironment.
        # detect()`) and only if the caller hasn't already supplied `plugins`
        # themselves, so this doesn't affect non-Slurm (local/dev) runs or
        # override an explicit `--trainer.plugins`.
        plugins = kwargs.pop("plugins", None)
        if plugins is None and SLURMEnvironment.detect():
            plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]

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
            plugins=plugins,
            **kwargs,
        )
