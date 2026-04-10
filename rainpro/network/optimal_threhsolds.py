import torch
from einops import rearrange, repeat
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rainpro.loss.ordinal_consistent import Bucket, Threshold, _sevir_buckets
from rainpro.metrics.csi import CriticalSuccessIndex
from rainpro.modules.utils import EvalOutputs


class OptimalThresholds(Callback):
    def __init__(
        self,
        output_dir,
        threshold_module: Threshold,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.buckets = _sevir_buckets(normalize=False)
        self.n_lead_times = threshold_module.thresholds.size(1)

    def on_predict_start(self, trainer, pl_module):
        return self.update_thresholds(trainer, pl_module)

    def on_test_start(self, trainer, pl_module):
        return self.update_thresholds(trainer, pl_module)

    def update_thresholds(self, trainer, pl_module):
        print("Updating best thresholds")
        best_thresholds_path = self.output_dir / "best_thresholds.pt"
        if best_thresholds_path.exists():
            print(f"Best thresholds already stored at {best_thresholds_path}")
            best_thresholds = torch.load(best_thresholds_path)
        else:
            best_thresholds = calculate_optimal_thresholds(
                trainer,
                pl_module,
                self.buckets,
                self.n_lead_times,
            )
            torch.save(best_thresholds, best_thresholds_path)

        # Update current thresholds with the optimal thresholds
        best_thresholds = rearrange(best_thresholds, "t c -> 1 t c 1 1")
        sd = pl_module.state_dict()
        for key in pl_module.state_dict().keys():
            if key == "model.thresholds.thresholds":
                sd[key] = best_thresholds
        pl_module.load_state_dict(sd)


def calculate_optimal_thresholds(
    trainer: Trainer,
    module: LightningModule,
    buckets: list[Bucket],
    n_lead_times: int,
):
    min_threshold = 0.01
    max_threshold = 0.6
    num_thresholds = 60
    num_samples = -1

    print(f"Considering {num_thresholds} thresholds in {min_threshold, max_threshold}")

    trainer.datamodule.setup("validate")
    if num_samples != -1:
        val_dataset = trainer.datamodule.val_dataset
        indices = torch.randperm(len(val_dataset))[:num_samples]
        subset = Subset(val_dataset, indices.tolist())
        dataloader = DataLoader(
            subset,
            batch_size=trainer.datamodule.batch_size,
            num_workers=trainer.datamodule.eval_workers,
            pin_memory=True,
        )
    else:
        dataloader = trainer.datamodule.val_dataloader()

    thresholds = torch.linspace(min_threshold, max_threshold, num_thresholds)
    bucket_vals = [b.min for b in buckets]
    threshold_csi = {}
    for threshold in thresholds:
        threshold_csi[threshold] = CriticalSuccessIndex(
            thresholds=bucket_vals,
            num_lead_times=n_lead_times,
            average_time=False,
        ).to(module.device)

    bucket_vals = rearrange(
        torch.tensor(bucket_vals), "c -> 1 1 c 1 1", c=len(buckets)
    ).to(module.device)
    for batch in tqdm(dataloader, desc="Calculating best thresholds"):
        for k, t in batch.items():
            batch[k] = t.to(module.device)
        eval_outputs: EvalOutputs = module.predict_step(
            batch,
            None,
            need_target=True,
            need_probs=True,
        )

        targets = eval_outputs.target
        probs = eval_outputs.probs
        assert probs is not None

        probs_gte = 1 - probs

        for threshold, csi in threshold_csi.items():
            threshold = repeat(
                torch.tensor([threshold]),
                "1 -> 1 t c 1 1",
                t=n_lead_times,
                c=len(buckets),
            ).to(module.device)
            val_candidates = (probs_gte > threshold) * bucket_vals
            val_output = torch.max(val_candidates, dim=2, keepdim=True).values
            csi.update(EvalOutputs(val_output, targets, None, None))

    results = []
    for threshold, csi in threshold_csi.items():
        result = csi.compute()
        results.append(result)

    results = torch.stack(results).cpu()  # (thresholds, buckets, lead_times)
    best_thresholds = torch.argmax(results, dim=0)  # (buckets, lead_times)
    best_thresholds = torch.tensor(thresholds)[best_thresholds]
    best_thresholds = rearrange(best_thresholds, "c t -> t c")

    # Consistent thresholds, avoid overfitting
    best_thresholds = torch.minimum(
        best_thresholds, best_thresholds.cummin(dim=0).values
    )

    trainer.datamodule.setup("test")
    return best_thresholds
