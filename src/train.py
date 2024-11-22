import os
import hydra
from omegaconf import DictConfig
import mlflow
import lightning as pl
import torch


from src.datasets.cgan_dataset import GAN_Dataset
from src.arch.proper_cgan.model import GAN_Model, GNet

from torch.utils.data import DataLoader


# Enable autologging
mlflow.pytorch.autolog()


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def train(cfg: DictConfig):
    # Create a new MLflow Experiment
    mlflow.set_experiment("Model training with MLFlow")

    dataset = GAN_Dataset(cfg.train_path, split="train")

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_G = GNet(device, optimizer="Adam", body="resnet34")
    GAN_model = GAN_Model(model_G.G_net, lr_G=0.0004, lr_D=0.0004, beta1=0.5, beta2=0.999, lamda=100.)

    print("Starting run...")

    with mlflow.start_run() as run:
        GAN_model.train_model(train_loader, epochs=200)

if __name__ == "__main__":
    train()
