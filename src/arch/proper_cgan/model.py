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
class Discriminator(pl.LightningModule):
    """The Discriminator. Uses PatchGAN achitecture."""

    def __init__(
        self,
        lr=0.0004,
        beta1=0.5,
        beta2=0.999,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.configure_model()

    def configure_model(self):
        self.model = self._build()
        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build(self, n_input=3, n_filters=64, sublayers=3) -> nn.Sequential:
        """Build the Discriminator. Uses PatchGAN architecture.

        Args:
            n_input (int): Number of input channels. Default to 3.
            n_filters (int): Number of filters to use in sublayers. Defaults to 64.
            sublayers (int): Number of deeper sublayers. Defaults to 3.

        Returns:
            nn.Sequential: Model
        """
        model = [
            self._make_sublayer(
                n_input,
                n_filters,
                norm=False,
            )
        ]
        model += [
            self._make_sublayer(
                n_filters * 2**i,
                n_filters * 2 ** (i + 1),
                s=1 if i == (sublayers - 1) else 2,
            )
            for i in range(sublayers)
        ]
        model += [
            self._make_sublayer(
                n_filters * 2**sublayers,
                1,
                s=1,
                norm=False,
                act=False,
            )
        ]
        model += [nn.Sigmoid()]  # TODO Experiment and remove if needed

        return nn.Sequential(*model)

    def _make_sublayer(
        self, ni, nf, k=4, s=2, p=1, norm=True, act=True
    ) -> nn.Sequential:
        """Make a sublayer

        Args:
            ni (int): Number of input channels.
            nf (int): Number of filters.
            k (int, optional): Kernel size. Defaults to 4.
            s (int, optional): Stride. Defaults to 2.
            p (int, optional): Padding. Defaults to 1.
            norm (bool, optional): Whether to add a norm layer. Defaults to True.
            act (bool, optional): Whether to add an activation layer. Defaults to True.

        Returns:
            nn.Sequential: PatchGAN
        """
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )


# Generator
class Generator(pl.LightningModule):
    """Generator model. Uses U-net with Resnet34 at its core."""

    def __init__(
        self,
        lr=0.0004,
        beta1=0.5,
        beta2=0.999,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.configure_model()

        self.criterion = nn.L1Loss()

    def configure_model(self):
        self.model = self._build()
        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _build(self, n_input=1, n_output=2, size=256) -> DynamicUnet:
        """Build the Generator

        Args:
            n_input (int, optional): Number of input channels. Defaults to 1 for L.
            n_output (int, optional): Number of output channels. Defaults to 2 for a,b.
            size (int, optional): Image size. Defaults to 256.

        Returns:
            DynamicUnet: Model
        """
        body_model = resnet34()
        backbone = create_body(body_model, pretrained=True, n_in=n_input, cut=-2)
        model = DynamicUnet(backbone, n_output, (size, size)).to(self.device)
        return model

    def forward(self, X):
        return self.model(X)

    # def training_step(self, batch):
    #     L, ab = batch["L"].to(self.device), batch["ab"].to(self.device)

    #     preds = self(L)
    #     loss = self.criterion(preds, ab)

    #     self.log("loss_pretrain_L1", loss, prog_bar=True)

    #     return loss

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )


class ColorizationGAN(pl.LightningModule):
    def __init__(self, lamda=100.0):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.G_net = Generator()
        self.D_net = Discriminator()

        self.GANcriterion = GANLoss()
        self.L1criterion = nn.L1Loss()

    def forward(self, L):
        return self.G_net(L)

    def training_step(self, batch):
        L, ab = batch[:, [0], :, :], batch[:, [1, 2], :, :]
        
        fake_color = self(L)

        opt_D, opt_G = self.optimizers()

        # Train Discriminator
        opt_D.zero_grad()

        fake_image = torch.cat([L, fake_color.detach()], dim=1)
        fake_preds = self.D_net(fake_image)
        loss_D_fake = self.GANcriterion(fake_preds, False)

        real_image = torch.cat([L, ab], dim=1)
        real_preds = self.D_net(real_image)
        loss_D_real = self.GANcriterion(real_preds, True)

        loss_D = (loss_D_fake + loss_D_real) / 2

        self.manual_backward(loss_D)
        opt_D.step()

        self.log_dict(
            {
                "loss_D": loss_D,
                "loss_D_fake": loss_D_fake,
                "loss_D_real": loss_D_real,
            },
            prog_bar=True,
        )

        # Train Generator
        opt_G.zero_grad()

        fake_image = torch.cat([L, fake_color], dim=1)
        fake_preds = self.D_net(fake_image)
        loss_G_GAN = self.GANcriterion(fake_preds, True)
        loss_G_L1 = self.L1criterion(fake_color, ab)

        loss_G = loss_G_GAN + loss_G_L1 * self.hparams.lamda

        self.log_dict(
            {
                "loss_G": loss_G,
                "loss_G_GAN": loss_G_GAN,
                "loss_G_L1": loss_G_L1,
            },
            prog_bar=True,
        )

        self.manual_backward(loss_G)
        opt_G.step()

    def configure_optimizers(self):
        opt_D = self.D_net.configure_optimizers()
        opt_G = self.G_net.configure_optimizers()
        return [opt_D, opt_G]
