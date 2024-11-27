# Import PyTorch Lightning
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.arch.proper_cgan.dataset import GAN_Dataset


class GANDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir_train,
        data_dir_test,
        batch_size,
        num_workers,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir_train = data_dir_train
        self.data_dir_test = data_dir_test

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            entire_dataset = GAN_Dataset(self.data_dir_train)

            self.train_ds, self.val_ds = random_split(entire_dataset, [0.8, 0.2])

        if stage == "test" or stage == "predict" or stage is None:
            self.test_ds = GAN_Dataset(self.data_dir_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
