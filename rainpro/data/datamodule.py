import datetime
from typing import Any

from einops.layers.torch import Rearrange
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from rainpro.data.dataset import SEVIRDataset


class SEVIRDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        frames_in: int,
        frames_out: int,
        img_size: int,
        stride: int,
        batch_size: int = 1,
        eval_batch_size: int | None = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = data_root
        self.batch_size = batch_size
        self.eval_batch_size = (
            eval_batch_size if eval_batch_size is not None else batch_size
        )
        self.num_workers = num_workers
        self.seq_len = frames_in + frames_out
        self.stride = stride
        self.train_valid_split = datetime.datetime(2019, 1, 1)
        self.valid_test_split = datetime.datetime(2019, 10, 1)

        self.transforms = {
            "vil": transforms.Compose(
                [
                    Rearrange("h w t -> t h w"),
                    transforms.Resize((img_size, img_size)),
                ]
            )
        }

    def create_dataset(
        self,
        start_date: datetime.datetime | None,
        end_date: datetime.datetime | None,
    ) -> SEVIRDataset:
        return SEVIRDataset(
            dataset_dir=self.data_root,
            data_types=["vil"],
            transforms=self.transforms,
            seq_len=self.seq_len,
            raw_seq_len=49,
            stride=self.stride,
            start_date=start_date,
            end_date=end_date,
            verbose=False,
        )

    def state_dict(self) -> dict[str, Any]:
        return {"transforms": self.transforms}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.transforms = state_dict["transforms"]

    def train_dataloader(self):
        train_dataset = self.create_dataset(None, self.train_valid_split)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        val_dataset = self.create_dataset(self.train_valid_split, self.valid_test_split)
        return DataLoader(
            val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )

    def test_dataloader(self):
        test_dataset = self.create_dataset(self.valid_test_split, None)
        return DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
