import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from torch.utils.data import random_split

from src.arch.proper_cgan.dataset import GAN_Dataset

# Import PyTorch Lightning
import pytorch_lightning as pl



class GanDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage): # https://youtu.be/NjF1ZpRO4Ws?si=vmrpGVxMMUIt6dhC 4:00 что делать с кастомным датасетом
        
        entire_dataset = GAN_Dataset(self.data_dir, split="train")
        #entire_dataset = datasets.MNIST(root=self.data_dir, train=True, transform=transforms.ToTensor(), download=False)

        self.train_ds, self.val_ds = random_split(entire_dataset, [0.8, 0.2])

        #self.test_ds = datasets.MNIST(root=self.data_dir, train=False, transform=transforms.ToTensor(), download=False)



    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
        

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
