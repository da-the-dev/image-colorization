import os
import hydra
from matplotlib import pyplot as plt
import mlflow
import lightning as pl

import numpy as np
from omegaconf import DictConfig
import torch

from src.datasets.cgan_dataset import GAN_Dataset
from src.arch.proper_cgan.model import ColorizationGAN, Generator

from torch.utils.data import DataLoader

from skimage.color import lab2rgb

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

# Enable autologging
mlflow.pytorch.autolog()

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in mlflow.MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


def visualize(model, image: np.ndarray, save=True):
    L, a, b = image
    
    with torch.no_grad():
        fake_color = model.forward(L)
    
    fake_color = model.fake_color.detach()
    # real_color = 
    
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    fig
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def train(cfg: DictConfig):
    # Create a new MLflow Experiment
    mlflow.set_experiment("Model training with MLFlow")
    
    
    model = ColorizationGAN()
    trainer = pl.Trainer(max_epochs=cfg.model.epochs)

    dataset = GAN_Dataset(cfg.train_path, split="train")

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    
    print("Starting run...")

    with mlflow.start_run() as run:
        trainer.fit(model, train_loader)
        
    print(print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id)))
    
    
    
    # mlflow.log_image()
    
if __name__ == "__main__":
    train()