from datetime import timedelta

from lightning import Callback
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from rainpro.callbacks.log_animations import LogAnimations
from rainpro.callbacks.log_plots import LogPlots


def create_callbacks(
    ckpt_path: str,
    early_stopping_patience: int | None,
    num_animations: int | None,
    animation_bounds: list[float] | None = None,
    checkpoint_interval_minutes: float | None = 30,
) -> list[Callback]:
    callbacks = [
        LogPlots(),
        LearningRateMonitor(),
        # Model selection: the best epoch by val/loss. Only ever written when
        # validation runs, i.e. at epoch end.
        ModelCheckpoint(
            dirpath=ckpt_path,
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_on_train_epoch_end=False,
            save_last=False,
            enable_version_counter=False,
        ),
    ]

    if checkpoint_interval_minutes is not None:
        # Resume point on a wall-clock interval, deliberately independent of
        # the validation cadence. "best" above can only appear after a full
        # epoch, so on a run whose epoch is longer than the SLURM time limit,
        # a job that hits the wall leaves *nothing* to resume from -- and
        # `RainProTrainer(auto_requeue=False)` means SLURM won't
        # checkpoint-and-resubmit on our behalf either. Passing only
        # `train_time_interval` (no every_n_epochs/every_n_train_steps) makes
        # this fire on that interval and nothing else; `monitor=None` +
        # `save_top_k=1` keeps just the most recent, overwriting `last.ckpt`.
        # Resume with `--ckpt_path <root_dir>/<run_name>/checkpoints/last.ckpt`.
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_path,
                filename="last",
                train_time_interval=timedelta(minutes=checkpoint_interval_minutes),
                save_top_k=1,
                enable_version_counter=False,
            )
        )

    if num_animations is not None and num_animations > 0:
        callbacks.append(
            LogAnimations(num_animations=num_animations, bounds=animation_bounds),
        )

    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss", mode="min", patience=early_stopping_patience
            )
        )

    return callbacks
