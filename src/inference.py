import os
import hydra
from matplotlib import pyplot as plt
import mlflow
import numpy as np
from omegaconf import DictConfig
import torch

from src.arch.proper_cgan.pl_dataset import GanDataModule
from src.arch.proper_cgan.utils import lab2rgb_denormalize


def load_model(uri):
    return mlflow.pytorch.load_model(uri)


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg: DictConfig):
    datamodule = GanDataModule(cfg.train_path, cfg.batch_size, os.cpu_count())
    datamodule.setup("predict")
    val_batch = np.stack([lab2rgb_denormalize(img) for img in next(iter(datamodule.val_dataloader()))[:5]])

    uri = "MODEL URI"
    model = load_model(uri)
    model.eval()

    # LAB, normalized, tensor
    L = val_batch[:, [0], :, :].to("cuda")
    fake_ab = model(L)

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
        ax.imshow(val_batch[i])
        ax.axis("off")

    # Convert the Matplotlib figure to a PIL Image
    fig.tight_layout()

    print("saving...")
    fig.savefig("img.png")


if __name__ == "__main__":
    main()
