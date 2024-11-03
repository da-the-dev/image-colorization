import lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim

from fastai.vision.all import *
from fastai.vision.models.unet import DynamicUnet

from torchvision.models.resnet import resnet34


# GAN Loss
class GANLoss(nn.Module):
    def __init__(self, gan_mode="vanilla", real_label=0.9, fake_label=0.1):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label))
        self.register_buffer("fake_label", torch.tensor(fake_label))
        if gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [
            self.get_layers(
                num_filters * 2**i,
                num_filters * 2 ** (i + 1),
                s=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]
        model += [
            self.get_layers(num_filters * 2**n_down, 1, s=1, norm=False, act=False)
        ]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ColorizationGAN(pl.LightningModule):
    def __init__(
        self, G_net=None, lr_G=0.0004, lr_D=0.0004, beta1=0.5, beta2=0.999, lamda=100.0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.G_net = G_net
        self.D_net = Discriminator(input_c=3, n_down=3, num_filters=64)
        self.GANcriterion = GANLoss(gan_mode="vanilla")
        self.L1criterion = nn.L1Loss()

    def forward(self, L):
        return self.G_net(L)

    def training_step(self, batch, batch_idx, optimizer_idx):
        L, ab = batch["L"], batch["ab"]
        fake_color = self(L)

        # Train Discriminator
        if optimizer_idx == 0:
            fake_image = torch.cat([L, fake_color.detach()], dim=1)
            fake_preds = self.D_net(fake_image)
            loss_D_fake = self.GANcriterion(fake_preds, False)

            real_image = torch.cat([L, ab], dim=1)
            real_preds = self.D_net(real_image)
            loss_D_real = self.GANcriterion(real_preds, True)

            loss_D = (loss_D_fake + loss_D_real) * 0.5

            self.log("loss_D", loss_D, prog_bar=True)
            self.log("loss_D_fake", loss_D_fake, prog_bar=True)
            self.log("loss_D_real", loss_D_real, prog_bar=True)

            return loss_D

        # Train Generator
        if optimizer_idx == 1:
            fake_image = torch.cat([L, fake_color], dim=1)
            fake_preds = self.D_net(fake_image)
            loss_G_GAN = self.GANcriterion(fake_preds, True)
            loss_G_L1 = self.L1criterion(fake_color, ab) * self.hparams.lamda
            loss_G = loss_G_GAN + loss_G_L1

            self.log("loss_G", loss_G, prog_bar=True)
            self.log("loss_G_GAN", loss_G_GAN, prog_bar=True)
            self.log("loss_G_L1", loss_G_L1, prog_bar=True)

            return loss_G

    def configure_optimizers(self):
        opt_D = optim.Adam(
            self.D_net.parameters(),
            lr=self.hparams.lr_D,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        opt_G = optim.Adam(
            self.G_net.parameters(),
            lr=self.hparams.lr_G,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        return [opt_D, opt_G]


class Generator(pl.LightningModule):
    def __init__(self, device, lr=0.0004):
        self.device = device
        self.G_net = self.build_G_net(n_input=1, n_output=2, size=256)
        self.criterion = nn.L1Loss()

    def build_G_net(self, n_input=1, n_output=2, size=256):
        body_model = resnet34()
        backbone = create_body(body_model, pretrained=True, n_in=n_input, cut=-2)
        G_net = DynamicUnet(backbone, n_output, (size, size)).to(self.device)
        return G_net

    def training_step(self, batch):
        L, ab = batch["L"].to(self.device), batch["ab"].to(self.device)

        preds = self.G_net(L)
        loss = self.criterion(preds, ab)

        self.log("loss_pretrain_L1", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.G_net.parameters(),
            lr=self.hparams.lr,
        )

        return optimizer
