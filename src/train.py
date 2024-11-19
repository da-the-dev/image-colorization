import os
import hydra
from omegaconf import DictConfig
import mlflow
import lightning as pl


from src.datasets.cgan_dataset import GAN_Dataset
from src.arch.proper_cgan.model import ColorizationGAN

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

    print("Loading test images...")
    test_images = next(iter(train_loader))
    print("Loaded test images!")

    model = ColorizationGAN(test_images, lamda=cfg.model.lamda)
    trainer = pl.Trainer(max_epochs=cfg.model.epochs)

    print("Starting run...")

    with mlflow.start_run() as run:
        print("Started fit...")
        trainer.fit(model, train_loader)
        print("Fitted!")


if __name__ == "__main__":
    train()
