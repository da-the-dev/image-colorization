import torch
import torch.nn as nn
import torch.optim as optim
import lightning as pl
from tqdm import tqdm
from fastai.vision.all import *
from fastai.vision.models.unet import DynamicUnet
from torchvision.models.resnet import resnet34

from src.arch.proper_cgan.losses import GANLoss


class Generator(pl.LightningModule):
    def __init__(
        self,
        lr=0.0004,
        beta1=0.9,
        beta2=0.999,
    ):
        self.save_hyperparameters()
        self.configure_model()
        self.criterion = nn.L1Loss()

    def configure_model(self):
        self.model = self._build()

    def _build(self, n_input=1, n_output=2, size=256):
        body_model = resnet34()
        backbone = create_body(body_model, pretrained=True, n_in=n_input, cut=-2)
        model = DynamicUnet(backbone, n_output, (size, size)).to(self.device)
        return model

    def forward(self, x):
        return self.model(x)

    def train_step(self, batch):
        L = batch[:, [0], :, :].to(self.device)
        ab = batch[:, [1, 2], :, :].to(self.device)

        preds = self(L)

        loss = self.criterion(preds, ab)

        return loss

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(),
            self.hparams.lr,
            (self.hparams.beta1, self.hparams.beta2),
        )
