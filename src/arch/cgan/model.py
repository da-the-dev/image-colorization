import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from fastai.vision.all import *
from fastai.vision.models.unet import DynamicUnet

from torchvision.models.resnet import resnet34

from src.arch.cgan.utils import *

from torch.utils.tensorboard import SummaryWriter


# Class for initializing Generator model
class GNet:
    def __init__(self, device, G_net=None, optimizer="Adam", body="resnet34"):
        self.device = device

        if G_net is None:
            self.G_net = self.build_G_net(n_input=1, n_output=2, size=256, body=body)
        else:
            self.G_net = G_net

        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.G_net.parameters(), lr=0.0004)

        self.criterion = nn.L1Loss()

    def build_G_net(self, n_input=1, n_output=2, size=256, body="resnet34"):
        if body == "resnet34":
            body_model = resnet34()
        backbone = create_body(body_model, pretrained=True, n_in=n_input, cut=-2)
        G_net = DynamicUnet(backbone, n_output, (size, size)).to(self.device)
        return G_net

    def pretrain(self, train_dl, epochs):
        writer = SummaryWriter(comment="cgan_pretrain")

        for epoch in range(epochs):
            losses = []
            with tqdm(train_dl, unit="batch") as tq:
                for data in tq:
                    L, ab = data["L"].to(self.device), data["ab"].to(self.device)
                    
                    preds = self.G_net(L)
                    loss = self.criterion(preds, ab)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    tq.set_postfix(loss=loss.item())
                    
                    losses.append(loss)
                
            average_loss = sum(losses) / len(losses)
            writer.add_scalar("Loss/pretrain", average_loss, epoch)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"L1 Loss: {average_loss:.2f}")
            
        writer.flush()
        writer.close()

    def get_model(self):
        return self.G_net

    def save_model(self, path="generator.pt"):
        torch.save(self.G_net.state_dict(), path)

    def load_model(self, path="generator.pt"):
        self.G_net.load_state_dict(torch.load(path), map_location=self.device)


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


# GAN Model
class GAN_Model(nn.Module):
    def __init__(
        self, G_net, lr_G=0.0004, lr_D=0.0004, beta1=0.5, beta2=0.999, lamda=100.0
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lamda = lamda

        self.G_net = G_net.to(self.device)
        self.D_net = init_model(
            Discriminator(input_c=3, n_down=3, num_filters=64), self.device
        )
        self.GANcriterion = GANLoss(gan_mode="vanilla").to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.G_net.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.D_net.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data["L"].to(self.device)
        self.ab = data["ab"].to(self.device)

    def forward(self):
        self.fake_color = self.G_net(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.D_net(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.D_net(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.D_net(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lamda
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.D_net.train()
        self.set_requires_grad(self.D_net, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.G_net.train()
        self.set_requires_grad(self.D_net, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

    def train_model(self, train_dl, epochs, display_every=200):
        # пока что ток трейн (лосс на трейне тоже не плохо)

        writer = SummaryWriter(comment="cgan_train")
        for epoch in range(epochs):
            loss_meter_dict = create_loss_meters()

            for i, data in enumerate(tqdm(train_dl)):
                self.setup_input(data)
                self.optimize()

                writer.add_scalars(
                    "Loss/train",
                    {
                        "loss_D_fake": self.loss_D_fake,
                        "loss_D_real": self.loss_D_real,
                        "loss_D": self.loss_D,
                        "loss_G_GAN": self.loss_G_GAN,
                        "loss_G_L1": self.loss_G_L1,
                        "loss_G": self.loss_G,
                    },
                )

                update_losses(self, loss_meter_dict, count=data["L"].size(0))

                if i % display_every == 0:
                    print(f"\nEpoch {epoch+1}/{epochs}")
                    print(f"Iteration {i}/{len(train_dl)}")
                    log_results(loss_meter_dict)
                    visualize(self, data, save=True)
        writer.flush()
        writer.close()

    def save_model(self, path="model.pt"):
        torch.save(self.state_dict(), path)
