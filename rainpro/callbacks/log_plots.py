import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from typing_extensions import Literal

import wandb
from rainpro.metrics.csi import CriticalSuccessIndex
from rainpro.metrics.probabilistic import ReliabilityAccumulator


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

        log_dict = {"global_step": trainer.global_step}

        for _, metric in metrics.items():
            if isinstance(metric, CriticalSuccessIndex):
                self._log_csi(log_dict, split, metric)
            elif isinstance(metric, ReliabilityAccumulator):
                self._log_reliability_table(log_dict, split, metric)
            elif hasattr(metric, "full"):
                self._log_pooled(log_dict, split, metric)

        logger.experiment.log(log_dict)

    def _log_csi(self, log_dict: dict, split: str, metric: CriticalSuccessIndex):
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

    def _log_pooled(self, log_dict: dict, split: str, metric):
        """Generic per-lead-time plotting for any metric exposing
        `full() -> dict[str, Tensor]` of 1D `[T]` or 2D `[K, T]` tensors --
        mirrors `_log_csi`'s three plot shapes for the 2D (threshold-like `K`)
        case, or a single per-lead-time line for the 1D case (e.g. CRPS,
        MAE, MSE)."""
        for name, tensor in metric.full().items():
            if tensor.ndim == 1:
                x_values = list(range(1, tensor.shape[0] + 1))
                y_values = tensor.cpu().tolist()
                table = wandb.Table(
                    data=[[x, y] for x, y in zip(x_values, y_values)],
                    columns=["Lead Time", name],
                )
                log_dict[f"{split}/{name}"] = wandb.plot.line(
                    table, "Lead Time", name, title=name
                )
                continue

            labels = getattr(metric, "labels", None) or [str(i) for i in range(tensor.shape[0])]

            mean_over_k = tensor.mean(dim=0)
            x_values = list(range(1, mean_over_k.shape[0] + 1))
            y_values = mean_over_k.cpu().tolist()
            table_mean = wandb.Table(
                data=[[x, y] for x, y in zip(x_values, y_values)],
                columns=["Lead Time", name],
            )
            log_dict[f"{split}/{name}_mean"] = wandb.plot.line(
                table_mean, "Lead Time", name, title=f"{name}-m"
            )

            for i, label in enumerate(labels):
                y_values = tensor[i].cpu().tolist()
                x_values = list(range(1, len(y_values) + 1))
                table = wandb.Table(
                    data=[[x, y] for x, y in zip(x_values, y_values)],
                    columns=["Lead Time", name],
                )
                log_dict[f"{split}/{name}/{label}"] = wandb.plot.line(
                    table, "Lead Time", name, title=f"{name}-{label}"
                )

            avg_values = [tensor[i].mean().item() for i in range(tensor.shape[0])]
            table_k = wandb.Table(
                data=[[lab, v] for lab, v in zip(labels, avg_values)],
                columns=["Threshold", name],
            )
            log_dict[f"{split}/{name}_thresholds"] = wandb.plot.line(
                table_k, "Threshold", name, title=f"{name} per Threshold"
            )

    def _log_reliability_table(self, log_dict: dict, split: str, metric: ReliabilityAccumulator):
        table = wandb.Table(
            columns=["bucket_dbz", "lead_time", "bin", "mean_pred", "obs_freq", "count"],
            data=metric.full_table(),
        )
        log_dict[f"{split}/reliability"] = table
