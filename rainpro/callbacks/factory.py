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
) -> list[Callback]:
    callbacks = [
        LogPlots(),
        LearningRateMonitor(),
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
