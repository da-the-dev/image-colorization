import torch
import torch.nn as nn
from fastai.vision.all import *
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet
import torch.optim as optim
from tqdm import tqdm


from torch import nn
from skimage.color import lab2rgb
import torch
import matplotlib.pyplot as plt
import numpy as np
import time


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def init_weights(net, init="norm", gain=0.02):
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

    net.apply(init_func)
    print(f"Initializing the model with {init} initialization")
    return net


def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model


# ====================


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {
        "loss_D_fake": loss_D_fake,
        "loss_D_real": loss_D_real,
        "loss_D": loss_D,
        "loss_G_GAN": loss_G_GAN,
        "loss_G_L1": loss_G_L1,
        "loss_G": loss_G,
    }


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    model.G_net.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.G_net.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap="gray")
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


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
        for itr in range(epochs):
            loss_meter = AverageMeter()
            for data in tqdm(train_dl):
                L, ab = data["L"].to(self.device), data["ab"].to(self.device)
                preds = self.G_net(L)
                loss = self.criterion(preds, ab)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_meter.update(loss.item(), L.size(0))

            print(f"Epoch {itr + 1}/{epochs}")
            print(f"L1 Loss: {loss_meter.avg:.5f}")

    def get_model(self):
        return self.G_net

    def save_model(self, path="generator.pt"):
        torch.save(self.G_net.state_dict(), path)

    def load_model(self, path="generator.pt"):
        self.G_net.load_state_dict(torch.load(path), map_location=self.device)


# GAN class implementation

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
        self.L = data[:, [0], :, :].to(self.device)
        self.ab = data[:, [1, 2], :, :].to(self.device)

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

    def train_model(self, train_dl, epochs, display_every=1):
        # пока что ток трейн (лосс на трейне тоже не плохо)
        for itr in range(epochs):
            loss_meter_dict = create_loss_meters()
            i = 0
            for data in tqdm(train_dl):
                self.setup_input(data)
                self.optimize()
                update_losses(self, loss_meter_dict, count=data[:, 0, :, :].size(0))
                i += 1
                
            if itr % display_every == 0:
                print(f"\nEpoch {itr+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict)
                visualize(self, data, save=True)

    def save_model(self, path="model.pt"):
        torch.save(self.state_dict(), path)
