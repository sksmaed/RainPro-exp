from typing import Literal

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection

from rainpro.data.rainpro8_datamodule import RainPro8DataModule
from rainpro.data.rainpro8_sources import context_sizes, tier_channels
from rainpro.loss.ordinal_consistent import OrdinalConsistentLoss, taiwan_dbz_buckets
from rainpro.metrics.contingency import ContingencyMetrics
from rainpro.metrics.csi import CriticalSuccessIndex
from rainpro.metrics.fss import FractionsSkillScore
from rainpro.metrics.probabilistic import CRPS, BrierScore, ReliabilityAccumulator
from rainpro.metrics.regression import LeadTimeMAEMSE
from rainpro.modules.utils import EvalOutputs, EvalRequest
from rainpro.network.rainpro8 import RainPro, StackTimeAndChannels

# GT is raw QPESUMS max dBZ (see `docs/rainpro_tw_implementation_notes.md`),
# not the paper's mm/h-converted thresholds -- these are threshold metrics
# (CSI/FSS/FBI/POD/FAR), invariant under the monotonic dBZ<->mm/h transform,
# so the *choice* of evaluation intensities is what's preserved, not the
# literal paper values.
CSI_THRESHOLDS_DBZ = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0]


def stack_sources(
    sources: dict, tier: str, tensors: dict[str, torch.Tensor]
) -> torch.Tensor | None:
    """Stack every source belonging to `tier` (B T C H W -> B (T C) H W each,
    concatenated on channels), matching `rainpro8.ipynb`. Returns `None` if no
    source is configured for `tier` (e.g. the 16km tier with `include_gfs=False`)."""
    stack = StackTimeAndChannels()
    matched = [stack(tensors[name]) for name, spec in sources.items() if spec.tier == tier]
    return torch.cat(matched, dim=1) if matched else None


class RainPro8Module(L.LightningModule):
    def __init__(
        self,
        data: RainPro8DataModule,
        max_epochs: int,
        skip_padding_4km: int = 32,
        dims: tuple[int, int, int, int] = (128, 256, 512, 128),
        cond_dim: int = 32,
        center_depth: int = 12,
        stochastic_depth_prob: float = 0.2,
        dropout: float = 0.1,
        resnet_depth: int = 2,
        lead_time_decay_ratio: float = 10.0,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.999),
    ):
        super().__init__()
        self.save_hyperparameters(ignore="data")

        self.sources = data.sources
        self.T_out = data.frames_out

        context_size_4km, context_size_8km = context_sizes(self.sources)
        in_dims = (
            tier_channels(self.sources, "4km"),
            tier_channels(self.sources, "8km"),
            tier_channels(self.sources, "16km"),
        )

        self.model = RainPro(
            T_out=self.T_out,
            in_dims=in_dims,
            context_size_4km=context_size_4km,
            context_size_8km=context_size_8km,
            skip_padding_4km=skip_padding_4km,
            dims=dims,
            cond_dim=cond_dim,
            center_depth=center_depth,
            stochastic_depth_prob=stochastic_depth_prob,
            dropout=dropout,
            resnet_depth=resnet_depth,
            buckets_fn=taiwan_dbz_buckets,
        )
        # Paper Sec. 3.3 / Sec. 4: lead time weights use a decay *ratio* (first vs.
        # last weight); rebuild the criterion with the requested ratio (default 2
        # in OrdinalConsistentLoss, 10 for the full RainPro-8 model).
        self.model.criterion = OrdinalConsistentLoss(
            self.model.out_channels,
            self.T_out,
            ratio=lead_time_decay_ratio,
            buckets_fn=taiwan_dbz_buckets,
        )

        self.val_metrics = create_metrics(self.T_out, "val")
        self.test_metrics = create_metrics(self.T_out, "test")

    def forward(
        self, batch: dict[str, torch.Tensor], eval_request: EvalRequest
    ) -> EvalOutputs | torch.Tensor:
        x_4km = stack_sources(self.sources, "4km", batch)
        x_8km = stack_sources(self.sources, "8km", batch)
        x_16km = stack_sources(self.sources, "16km", batch)

        target = batch["target_2km"] if (eval_request.need_target or eval_request.need_loss) else None

        return self.model.predict(x_4km, x_8km, x_16km, target, eval_request)

    def training_step(self, batch, batch_idx):
        loss = self(batch, EvalRequest(only_loss=True, need_loss=True))
        self.log("train/loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch, "test")

    def predict_step(
        self,
        batch,
        batch_idx,
        need_forecast: bool = False,
        need_target: bool = False,
        need_loss: bool = False,
        need_probs: bool = False,
    ) -> EvalOutputs:
        eval_request = EvalRequest(
            need_forecast=need_forecast,
            need_target=need_target,
            need_probs=need_probs,
            need_loss=need_loss,
        )
        return self(batch, eval_request)

    def _shared_eval(self, batch: dict, stage: Literal["val", "test"]) -> EvalOutputs:
        eval_request = EvalRequest(
            only_loss=False,
            need_forecast=True,
            need_target=True,
            need_loss=True,
            need_probs=True,
        )
        eval_outputs = self.forward(batch, eval_request)
        assert isinstance(eval_outputs, EvalOutputs)

        assert eval_outputs.loss is not None
        self.log(f"{stage}/loss", eval_outputs.loss, sync_dist=True)

        metrics = self.val_metrics if stage == "val" else self.test_metrics
        metrics.update(eval_outputs)
        self.log_dict(metrics)
        return eval_outputs

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Paper Sec. 4: AdamW, static lr=3e-4, weight_decay=0.1, betas=(0.9, 0.999).
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}


def create_metrics(num_lead_times: int, stage: str) -> MetricCollection:
    metric_dict = {
        "csi": CriticalSuccessIndex(
            num_lead_times=num_lead_times,
            thresholds=CSI_THRESHOLDS_DBZ,
        ),
        "contingency": ContingencyMetrics(
            num_lead_times=num_lead_times,
            thresholds=CSI_THRESHOLDS_DBZ,
        ),
        "fss": FractionsSkillScore(
            num_lead_times=num_lead_times,
            thresholds=CSI_THRESHOLDS_DBZ,
        ),
        "crps": CRPS(num_lead_times=num_lead_times),
        "brier": BrierScore(num_lead_times=num_lead_times),
        "reliability": ReliabilityAccumulator(num_lead_times=num_lead_times),
        "mae_mse": LeadTimeMAEMSE(num_lead_times=num_lead_times),
    }
    return MetricCollection(metric_dict, prefix=f"{stage}/")
