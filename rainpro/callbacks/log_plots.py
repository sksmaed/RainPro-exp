import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from typing_extensions import Literal

import wandb
from rainpro.metrics.csi import CriticalSuccessIndex


class LogPlots(pl.Callback):
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_plots("test", trainer, pl_module)

    @rank_zero_only
    def _log_plots(
        self,
        split: Literal["val", "test"],
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        logger: WandbLogger = trainer.logger
        metrics = getattr(pl_module, f"{split}_metrics", None)

        if metrics is None:
            return

        log_dict = {}
        log_dict["global_step"] = trainer.global_step

        for _, metric in metrics.items():
            if not isinstance(metric, CriticalSuccessIndex):
                continue

            all_csi = metric._compute(reduce_mean=False)

            # -----------------------
            # Mean CSI (over thresholds)
            # -----------------------
            mean_csi = all_csi.mean(dim=0)
            x_values = list(range(1, mean_csi.shape[0] + 1))
            y_values = mean_csi.cpu().tolist()

            table_mean = wandb.Table(
                data=[[x, y] for x, y in zip(x_values, y_values)],
                columns=["Lead Time", "CSI"],
            )

            log_dict[f"{split}/CSI_mean"] = wandb.plot.line(
                table_mean,
                "Lead Time",
                "CSI",
                title="CSI-m",
            )

            # -----------------------
            # Per-threshold CSI plots + averages
            # -----------------------
            for i in range(all_csi.shape[0]):
                label = metric.padded_names[i]

                y_values = all_csi[i].cpu().tolist()
                x_values = list(range(1, len(y_values) + 1))

                table = wandb.Table(
                    data=[[x, y] for x, y in zip(x_values, y_values)],
                    columns=["Lead Time", "CSI"],
                )

                log_dict[f"{split}/CSI/{label}"] = wandb.plot.line(
                    table,
                    "Lead Time",
                    "CSI",
                    title=f"CSI-{label}",
                )

        # -----------------------
        # CSI averaged over time vs threshold (line plot)
        # -----------------------
        thresholds = [str(t) for t in metric.padded_names]
        avg_csi_values = [all_csi[i].mean().item() for i in range(all_csi.shape[0])]

        table_thresh = wandb.Table(
            data=[[t, csi] for t, csi in zip(thresholds, avg_csi_values)],
            columns=["Threshold", "CSI"],
        )

        log_dict[f"{split}/CSI_thresholds"] = wandb.plot.line(
            table_thresh,
            "Threshold",
            "CSI",
            title="CSI per Threshold",
        )

        logger.experiment.log(log_dict)
