import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as pl
from tqdm import tqdm
from fastai.vision.all import *
from fastai.vision.models.unet import DynamicUnet
from torchvision.models.resnet import resnet34

from src.arch.proper_cgan.signature import signature
from src.arch.proper_cgan.utils import lab2rgb_denormalize
from src.arch.proper_cgan.losses import GANLoss


class Generator(pl.LightningModule):
    def __init__(
        self,
        lr=0.0004,
        beta1=0.9,
        beta2=0.999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["test_images"])
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

    def training_step(self, batch):
        L = batch[:, [0], :, :].to(self.device)
        ab = batch[:, [1, 2], :, :].to(self.device)

        preds = self(L)

        loss = self.criterion(preds, ab)

        self.log(
            "L1 loss for GNet during pretrain",
            loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        return loss

    def validation_step(self, batch, batchidx):
        L = batch[:, [0], :, :].to(self.device)
        ab = batch[:, [1, 2], :, :].to(self.device)

        preds = self(L)

        loss = self.criterion(preds, ab)

        self.log(
            "Validation L1 loss for GNet during pretrain",
            loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
        )

        # Save first batch for visualization
        if batchidx == 0:
            self.visualization_batch = batch[:5]

    def configure_optimizers(self):
        return optim.Adam(
            self.model.parameters(),
            self.hparams.lr,
            (self.hparams.beta1, self.hparams.beta2),
        )

    def on_validation_epoch_end(self):
        self.model.eval()
        # Setup imags for visualization
        images = self.visualization_batch
        images_cpu = np.stack(
            [lab2rgb_denormalize(img) for img in images.detach().cpu().numpy()]
        )

        # LAB, normalized, tensor
        L = images[:, [0], :, :].to(self.device)
        fake_ab = self.model(L)

        # LAB, normalized, numpy
        fake_images_cpu_lab = torch.concat([L, fake_ab], dim=1).detach().cpu().numpy()

        # rgb, denormalized, numpy
        fake_images_cpu_rgb = np.stack(
            [lab2rgb_denormalize(img) for img in fake_images_cpu_lab]
        )

        # Draw demo image
        fig = plt.figure(figsize=(15, 8))
        for i in range(5):
            ax = plt.subplot(3, 5, i + 1)
            ax.imshow(L[i][0].cpu(), cmap="gray")
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 5)
            ax.imshow(fake_images_cpu_rgb[i])
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 10)
            ax.imshow(images_cpu[i])
            ax.axis("off")

        # Convert the Matplotlib figure to a PIL Image
        fig.tight_layout()
        fig.canvas.draw()  # Draw the canvas

        # Convert to a NumPy array
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Log image to MLFlow
        mlflow.log_image(
            image_array,
            f"generated_images_gnet_pretrain_epoch_{self.current_epoch}.png",
        )
        self.model.train()


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
        registered_model_name,
        lr_G=0.0004,
        lr_D=0.0004,
        beta1_G=0.5,
        beta2_G=0.999,
        beta1_D=0.5,
        beta2_D=0.999,
        lamda=100.0,
        skip_epochs=200,
    ):
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(
            ignore=[
                "G_net",
                "test_images",
                "registered_model_name",
                "skip_epochs",
            ]
        )

        self.G_net = G_net
        self.D_net = Discriminator()

        self.GANcriterion = GANLoss(gan_mode="vanilla")
        self.L1criterion = nn.L1Loss()

        self.registered_model_name = registered_model_name
        self.skip_epochs = skip_epochs

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
            prog_bar=False,
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
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        # Backward for D_net
        self.manual_backward(loss_G)
        opt_G.step()
        self.untoggle_optimizer(opt_G)

    def validation_step(self, batch, batchidx):
        # Switch models to eval
        self.G_net.eval()

        # Split validation batch
        L = batch[:, [0], :, :]
        ab = batch[:, [1, 2], :, :]
        # Concat real image
        real_image = torch.cat([L, ab], dim=1)

        # Run generator to create a,b channels
        fake_color = self(L)
        # Concate fake image
        fake_image = torch.cat([L, fake_color], dim=1).detach()

        # Losses for D_net
        fake_preds = self.D_net(fake_image)
        loss_D_fake = self.GANcriterion(fake_preds, False)
        real_preds = self.D_net(real_image)
        loss_D_real = self.GANcriterion(real_preds, True)
        loss_D = (loss_D_fake + loss_D_real) / 2

        # Log losses
        self.log_dict(
            {
                "loss_D_fake_val": loss_D_fake.item(),
                "loss_D_real_val": loss_D_real.item(),
                "loss_D_val": loss_D.item(),
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        # Losses for G_net
        fake_preds = self.D_net(fake_image)
        loss_G_GAN = self.GANcriterion(fake_preds, True)
        loss_G_L1 = self.L1criterion(fake_color, ab) * self.hparams.lamda
        loss_G = loss_G_GAN + loss_G_L1

        # Log losses
        self.log_dict(
            {
                "loss_G_GAN_val": loss_G_GAN.item(),
                "loss_G_L1_val": loss_G_L1.item(),
                "loss_G_val": loss_G.item(),
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        # Switch models back to train
        self.G_net.train()

        # Save first batch for visualization
        if batchidx == 0 and self.current_epoch >= self.skip_epochs:
            self.visualiztion_batch = batch

    def configure_optimizers(self):
        opt_G = optim.Adam(
            self.G_net.parameters(),
            lr=self.hparams.lr_G,
            betas=(self.hparams.beta1_G, self.hparams.beta2_G),
        )
        opt_D = optim.Adam(
            self.D_net.parameters(),
            lr=self.hparams.lr_D,
            betas=(self.hparams.beta1_D, self.hparams.beta2_D),
        )
        return [opt_G, opt_D]

    def on_validation_epoch_end(self):
        # Skip image generation for the first couple epochs
        if self.current_epoch < self.skip_epochs:
            return

        images = self.visualiztion_batch
        images_cpu = np.stack(
            [lab2rgb_denormalize(img) for img in images.detach().cpu().numpy()]
        )

        # Switch generator to eval mode
        self.G_net.eval()
        self.D_net.eval()

        # LAB, normalized, tensor
        L = images[:, [0], :, :].to(self.device)
        fake_ab = self(L)

        # LAB, normalized, numpy
        fake_images_cpu_lab = torch.concat([L, fake_ab], dim=1).detach().cpu().numpy()

        # rgb, denormalized, numpy
        fake_images_cpu_rgb = np.stack(
            [lab2rgb_denormalize(img) for img in fake_images_cpu_lab]
        )

        # Draw demo image
        fig = plt.figure(figsize=(15, 8))
        for i in range(5):
            ax = plt.subplot(3, 5, i + 1)
            ax.imshow(L[i][0].cpu(), cmap="gray")
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 5)
            ax.imshow(fake_images_cpu_rgb[i])
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 10)
            ax.imshow(images_cpu[i])
            ax.axis("off")

        # Convert the Matplotlib figure to a PIL Image
        fig.tight_layout()
        fig.canvas.draw()  # Draw the canvas

        # Convert to a NumPy array
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        mlflow.log_image(
            image_array, f"generated_images_gan_epoch_{self.current_epoch}.png"
        )

        # Switch models back to train mode
        self.G_net.train()
        self.D_net.train()

    def on_train_epoch_end(self):
        if self.current_epoch >= self.skip_epochs:
            mlflow.pytorch.log_model(
                self,
                f"cgan_checkpoint_{self.current_epoch}",
                signature=signature,
                registered_model_name=self.registered_model_name,
            )
