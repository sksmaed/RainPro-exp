from typing import Literal

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import Metric, MetricCollection

from rainpro.data.datamodule import SEVIRDataModule
from rainpro.metrics.csi import CriticalSuccessIndex
from rainpro.modules.scheduler import get_cosine_schedule_with_warmup
from rainpro.modules.utils import EvalOutputs, EvalRequest
from rainpro.network.rainpro import RainPro


class SEVIRModule(L.LightningModule):
    def __init__(
        self,
        data: SEVIRDataModule,
        frames_in: int,
        frames_out: int,
        img_size: int,
        max_epochs: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="data")
        # `total_steps` is only needed by the LR scheduler in `configure_optimizers`,
        # which Lightning never calls for a test/predict-only run. Keep `data` and
        # compute it lazily there instead of eagerly here, so `test` doesn't have to
        # build the train dataloader (and thus open every h5 file referenced by the
        # train split) just to construct this module.
        self.data = data
        self.max_epochs = max_epochs

        self.frames_in = frames_in
        self.frames_out = frames_out

        self.model = RainPro(
            in_shape=(1, img_size, img_size),
            T_in=frames_in,
            T_out=frames_out,
        )

        thresholds = [16, 74, 133, 160, 181, 219]
        self.val_metrics = create_metrics(
            num_lead_times=frames_out, thresholds=thresholds, stage="val"
        )
        self.test_metrics = create_metrics(
            num_lead_times=frames_out, thresholds=thresholds, stage="test"
        )

    def forward(
        self, batch: dict, eval_request: EvalRequest
    ) -> EvalOutputs | torch.Tensor:
        inputs = batch["vil"][:, : self.frames_in, None]

        if eval_request.need_target or eval_request.need_loss:
            target = batch["vil"][:, self.frames_in :, None]
        else:
            target = None

        model_output = self.model.predict(inputs, target, eval_request)

        # Output loss
        if eval_request.only_loss:
            assert isinstance(model_output, torch.Tensor)
            return model_output

        # Output EvalOutput
        assert isinstance(model_output, EvalOutputs)

        if eval_request.need_forecast:
            model_output.forecast = model_output.forecast.clamp(0, 1.0) * 255

        if eval_request.need_target:
            assert model_output.target is not None
            model_output.target = model_output.target * 255

        if eval_request.need_loss:
            assert model_output.loss is not None

        if eval_request.need_probs:
            assert model_output.probs is not None

        return model_output

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

    def _shared_eval(
        self,
        batch: dict,
        stage: Literal["val", "test"],
    ) -> EvalOutputs:

        eval_request = EvalRequest(
            only_loss=False,
            need_forecast=True,
            need_target=True,
            need_loss=True,
            need_probs=False,
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
        trainable_params = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-4,
            betas=(0.90, 0.95),
            weight_decay=0.0,
        )

        total_steps = self.max_epochs * len(self.data.train_dataloader())
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "step",
            },
        }


def create_metrics(
    num_lead_times: int, thresholds: list[int], stage: str
) -> MetricCollection:
    metric_dict: dict[str, Metric | MetricCollection] = {
        "csi": CriticalSuccessIndex(
            num_lead_times=num_lead_times,
            thresholds=thresholds,
        )
    }
    metrics = MetricCollection(metric_dict, prefix=f"{stage}/")
    return metrics
