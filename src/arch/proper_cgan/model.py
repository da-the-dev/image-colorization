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
        super().__init__()
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


class Discriminator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.configure_model()

    def configure_model(self):
        self._build()
        self._init_model_weights()

    def forward(self, x):
        return self.model(x)

    def _build(self, input_c=3, num_filters=64, n_down=3):
        model = [self._get_layers(input_c, num_filters, norm=False)]
        model += [
            self._get_layers(
                num_filters * 2**i,
                num_filters * 2 ** (i + 1),
                s=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]
        model += [
            self._get_layers(num_filters * 2**n_down, 1, s=1, norm=False, act=False)
        ]
        self.model = nn.Sequential(*model)

    def _get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def _init_model_weights(self, init="norm", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and "Conv" in classname:
                if init == "norm":
                    nn.init.normal_(m.weight.data, mean=0.0, std=gain)
                elif init == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif "BatchNorm2d" in classname:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.model.apply(init_func)
        print(f"Initializing the model with {init} initialization")


class GAN(pl.LightningModule):
    def __init__(
        self,
        G_net,
        lr_G=0.0004,
        lr_D=0.0004,
        beta1=0.5,
        beta2=0.999,
        lamda=100.0,
    ):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["G_net"])

        self.G_net = G_net
        self.D_net = Discriminator()

        self.GANcriterion = GANLoss(gan_mode="vanilla")
        self.L1criterion = nn.L1Loss()

    def forward(self, L):
        return self.G_net(L)

    def training_step(self, batch):
        # Split traininig batch
        L = batch[:, [0], :, :]
        ab = batch[:, [1, 2], :, :]
        # Concat real image
        real_image = torch.cat([L, ab], dim=1)

        # Run generator to create a,b channels
        fake_color = self(L)
        # Concate fake image
        fake_image = torch.cat([L, fake_color], dim=1).detach()

        # Optimization
        opt_G, opt_D = self.optimizers()

        # Training Discriminator
        self.D_net.train()
        self.D_net.unfreeze()
        self.toggle_optimizer(opt_D)
        opt_D.zero_grad()

        # Losses for D_net
        fake_preds = self.D_net(fake_image)
        loss_D_fake = self.GANcriterion(fake_preds, False)
        real_preds = self.D_net(real_image)
        loss_D_real = self.GANcriterion(real_preds, True)
        loss_D = (loss_D_fake + loss_D_real) / 2

        # Log losses
        self.log_dict(
            {
                "loss_D_fake": loss_D_fake,
                "loss_D_real": loss_D_real,
                "loss_D": loss_D,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # Backward for D_net
        self.manual_backward(loss_D)
        opt_D.step()
        self.untoggle_optimizer(opt_D)

        # Training Generator
        self.G_net.train()
        self.D_net.freeze()
        self.toggle_optimizer(opt_G)
        opt_G.zero_grad()

        # Losses for G_net
        fake_preds = self.D_net(fake_image)
        loss_G_GAN = self.GANcriterion(fake_preds, True)
        loss_G_L1 = self.L1criterion(fake_color, ab) * self.hparams.lamda
        loss_G = loss_G_GAN + loss_G_L1

        # Log losses
        self.log_dict(
            {
                "loss_G_GAN": loss_G_GAN,
                "loss_G_L1": loss_G_L1,
                "loss_G": loss_G,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # Backward for D_net
        self.manual_backward(loss_G)
        opt_G.step()
        self.untoggle_optimizer(opt_G)

    def configure_optimizers(self):
        opt_G = optim.Adam(
            self.G_net.parameters(),
            lr=self.hparams.lr_G,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        opt_D = optim.Adam(
            self.D_net.parameters(),
            lr=self.hparams.lr_D,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        return [opt_G, opt_D]
