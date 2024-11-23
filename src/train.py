import os
import hydra
import mlflow
import lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.datasets.cgan_dataset import GAN_Dataset
from src.arch.proper_cgan.model import GAN, Generator

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

    print("Starting run...")
    with mlflow.start_run() as run:
        # TODO Test images for visualization

        G_net = Generator()
        print("Started generator pretrain...")
        pl.Trainer(max_epochs=cfg.model.pretrain_epochs).fit(G_net, train_loader)
        print("Generator pretrain completed!")

        print("Started GAN training...")
        GAN_model = GAN(G_net)
        pl.Trainer(max_epochs=cfg.model.epochs).fit(GAN_model, train_loader)
        print("GAN train completed!")


if __name__ == "__main__":
    train()
