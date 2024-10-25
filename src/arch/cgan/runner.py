import os
from omegaconf import DictConfig

from src.transform_data import transform_GAN_data
from src.prepare_data import get_data
from src.utils import device
import src.arch.cgan.model as cgan


def runner_cgan(cfg: DictConfig):
    train_paths, val_paths = get_data(num=10_000)

    train_dl = transform_GAN_data(
        train_paths,
        split="train",
        batch_size=cfg.model.batch_size,
    )
    val_dl = transform_GAN_data(
        val_paths,
        split="val",
        batch_size=cfg.model.batch_size,
    )

    model_G = cgan.GNet(
        device(),
        optimizer="Adam",
        body="resnet34",
    )
    model_G.pretrain(train_dl, epochs=5)
    model_G.save_model(path=os.path.join(cfg.model.models_dir, "gnet.pt"))

    GAN_model = cgan.GAN_Model(
        model_G.G_net,
        lr_G=0.0004,
        lr_D=0.0004,
        beta1=0.5,
        beta2=0.999,
        lamda=100.0,
    )
    GAN_model.train_model(train_dl, epochs=5)
    GAN_model.save_model(path=os.path.join(os.getcwd(), "cgan.pt"))
