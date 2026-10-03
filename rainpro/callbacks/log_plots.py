from dataclasses import dataclass, field

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from typing_extensions import Literal

import wandb
from rainpro.metrics.csi import CriticalSuccessIndex
from rainpro.metrics.probabilistic import ReliabilityAccumulator

# Baseline metric collections drawn on the SAME charts as the model's, as
# `<split>_<suffix>_metrics` attribute suffix -> legend name. A module without
# the attribute (or with it set to None) simply has no extra line.
BASELINES = {"optflow": "optical flow"}
MODEL_SERIES = "model"


@dataclass
class _Curve:
    title: str
    x_name: str
    y_name: str
    xs: list
    series: dict[str, list[float]] = field(default_factory=dict)


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
        curves: dict[str, _Curve] = {}

        sources = [(MODEL_SERIES, split, metrics)]
        for suffix, name in BASELINES.items():
            baseline = getattr(pl_module, f"{split}_{suffix}_metrics", None)
            if baseline is not None:
                sources.append((name, f"{split}_{suffix}", baseline))

        for series, table_prefix, collection in sources:
            for _, metric in collection.items():
                if isinstance(metric, CriticalSuccessIndex):
                    self._add_csi(curves, split, series, metric)
                elif isinstance(metric, ReliabilityAccumulator):
                    self._log_reliability_table(log_dict, table_prefix, metric)
                elif hasattr(metric, "full"):
                    self._add_pooled(curves, split, series, metric)

        for key, curve in curves.items():
            log_dict[key] = self._chart(curve)

        logger.experiment.log(log_dict)

    @staticmethod
    def _chart(curve: _Curve):
        """One line (the model alone): the plain `wandb.plot.line` these
        charts have always been. Model + baselines: one chart, one line each,
        so they compare directly."""
        if len(curve.series) == 1:
            (ys,) = curve.series.values()
            table = wandb.Table(
                data=[[x, y] for x, y in zip(curve.xs, ys)],
                columns=[curve.x_name, curve.y_name],
            )
            return wandb.plot.line(table, curve.x_name, curve.y_name, title=curve.title)
        return wandb.plot.line_series(
            xs=_numeric(curve.xs),
            ys=list(curve.series.values()),
            keys=list(curve.series),
            title=f"{curve.title} ({curve.y_name})",
            xname=curve.x_name,
        )

    @staticmethod
    def _add(curves: dict, key: str, series: str, title: str, x_name: str, y_name: str, xs, ys) -> None:
        curve = curves.setdefault(key, _Curve(title=title, x_name=x_name, y_name=y_name, xs=list(xs)))
        curve.series[series] = list(ys)

    def _add_csi(self, curves: dict, split: str, series: str, metric: CriticalSuccessIndex):
        all_csi = metric._compute(reduce_mean=False)  # [threshold, T]
        lead_times = list(range(1, all_csi.shape[1] + 1))

        # Mean CSI (over thresholds) vs lead time
        self._add(curves, f"{split}/CSI_mean", series, "CSI-m", "Lead Time", "CSI",
                  lead_times, all_csi.mean(dim=0).cpu().tolist())

        # Per-threshold CSI vs lead time
        for i, label in enumerate(metric.padded_names):
            self._add(curves, f"{split}/CSI/{label}", series, f"CSI-{label}", "Lead Time", "CSI",
                      lead_times, all_csi[i].cpu().tolist())

        # CSI averaged over time vs threshold
        self._add(curves, f"{split}/CSI_thresholds", series, "CSI per Threshold", "Threshold", "CSI",
                  [str(t) for t in metric.padded_names], all_csi.mean(dim=1).cpu().tolist())

    def _add_pooled(self, curves: dict, split: str, series: str, metric):
        """Generic per-lead-time curves for any metric exposing
        `full() -> dict[str, Tensor]` of 1D `[T]` or 2D `[K, T]` tensors --
        mirrors `_add_csi`'s three curve shapes for the 2D (threshold-like `K`)
        case, or a single per-lead-time line for the 1D case (e.g. CRPS,
        MAE, MSE)."""
        for name, tensor in metric.full().items():
            lead_times = list(range(1, tensor.shape[-1] + 1))
            if tensor.ndim == 1:
                self._add(curves, f"{split}/{name}", series, name, "Lead Time", name,
                          lead_times, tensor.cpu().tolist())
                continue

            labels = getattr(metric, "labels", None) or [str(i) for i in range(tensor.shape[0])]

            self._add(curves, f"{split}/{name}_mean", series, f"{name}-m", "Lead Time", name,
                      lead_times, tensor.mean(dim=0).cpu().tolist())
            for i, label in enumerate(labels):
                self._add(curves, f"{split}/{name}/{label}", series, f"{name}-{label}", "Lead Time", name,
                          lead_times, tensor[i].cpu().tolist())
            self._add(curves, f"{split}/{name}_thresholds", series, f"{name} per Threshold", "Threshold", name,
                      labels, tensor.mean(dim=1).cpu().tolist())

    def _log_reliability_table(self, log_dict: dict, prefix: str, metric: ReliabilityAccumulator):
        table = wandb.Table(
            columns=["bucket_dbz", "lead_time", "bin", "mean_pred", "obs_freq", "count"],
            data=metric.full_table(),
        )
        # Not "{prefix}/reliability": the module's `log_dict(metrics)` already logs
        # the scalar (ECE) under that key, and W&B can't chart a key holding
        # both a scalar and a Table.
        log_dict[f"{prefix}/reliability_table"] = table


def _numeric(labels: list) -> list:
    """Threshold labels ("20", "05", "5dBZ") -> numbers, so a multi-line chart
    gets a real x axis; falls back to positions for anything unparsable."""
    try:
        return [float(str(x).removesuffix("dBZ")) for x in labels]
    except ValueError:
        return list(range(len(labels)))
