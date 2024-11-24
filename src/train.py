import os
import hydra
import mlflow
import lightning as l
import mlflow.models
import mlflow.types
from omegaconf import DictConfig


from src.arch.proper_cgan.signature import signature
from src.arch.proper_cgan.pl_dataset import GANDataModule
from src.arch.proper_cgan.model import GAN, Generator

# Enable autologging
mlflow.pytorch.autolog(
    checkpoint=False,  # Skip checkpoining, no metrics, no need to save this info
    log_models=False,  # Skip logging models, we do it manually
)


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def train(cfg: DictConfig):
    # Create a new MLflow Experiment
    mlflow.set_experiment("Model training with MLFlow")

    dm = GANDataModule(
        data_dir_train=cfg.train_path,
        data_dir_test=cfg.test_path,
        batch_size=cfg.batch_size,
        num_workers=4,
    )
    # dm.setup()

    print("Starting run...")
    with mlflow.start_run() as run:
        G_net = Generator()

        print("Started generator pretrain...")
        pretrainer = l.Trainer(max_epochs=cfg.model.pretrain_epochs)
        pretrainer.fit(G_net, datamodule=dm)
        print("Generator pretrain completed!")

        mlflow.pytorch.log_model(
            pytorch_model=G_net,
            artifact_path="gnet",
            signature=signature,
            registered_model_name="U-net Generator (ResNet backbone)",
        )

        print("Started GAN training...")
        GAN_model = GAN(G_net)
        trainer = l.Trainer(max_epochs=cfg.model.epochs)
        trainer.fit(GAN_model, datamodule=dm)
        print("GAN train completed!")

        mlflow.pytorch.log_model(
            pytorch_model=GAN_model,
            artifact_path="gan",
            signature=signature,
            registered_model_name="Conditional GAN",
        )


if __name__ == "__main__":
    train()
