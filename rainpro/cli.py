from pathlib import Path

import torch
import yaml
from einops.layers.torch import Rearrange
from jsonargparse import ActionYesNo
from lightning.pytorch.cli import LightningCLI
from torchvision.transforms import Compose
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.transforms import Resize

from rainpro.data.datamodule import SEVIRDataModule
from rainpro.network.optimal_threhsolds import OptimalThresholds
from rainpro.trainer import RainProTrainer

torch.serialization.add_safe_globals([Compose, Rearrange, Resize, InterpolationMode])


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "run_name",
            type=str,
            help="Name of the run.",
        )
        parser.add_argument(
            "--thresholds",
            action=ActionYesNo,
            default=False,
            help="Whether to use optimized thresholds for testing.",
        )

        parser.link_arguments("run_name", "trainer.run_name")
        parser.link_arguments("data.frames_in", "model.init_args.frames_in")
        parser.link_arguments("data.frames_out", "model.init_args.frames_out")
        parser.link_arguments("data.img_size", "model.init_args.img_size")
        parser.link_arguments("trainer.max_epochs", "model.init_args.max_epochs")
        parser.link_arguments("data", "model.init_args.data", apply_on="instantiate")

    def before_fit(self):
        full_cfg = yaml.safe_load(self.parser.dump(self.config, skip_none=False))
        cfg = full_cfg[self.config["subcommand"]]
        path = Path(cfg["trainer"]["root_dir"]) / cfg["run_name"]
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "config.yml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def before_test(self):
        config = self.config[self.config["subcommand"]]
        if config["thresholds"]:
            print("Use optimized thresholds")
            output_dir = Path(config.get("ckpt_path")).parent
            self.trainer.callbacks.append(
                OptimalThresholds(
                    output_dir=output_dir,
                    threshold_module=self.model.model.thresholds,
                )
            )


def cli():
    CLI(
        datamodule_class=SEVIRDataModule,
        trainer_class=RainProTrainer,
        seed_everything_default=0,
    )
