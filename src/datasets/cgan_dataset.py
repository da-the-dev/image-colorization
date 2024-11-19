# Здесь будет класс, наследуемый от Datasets для работы с GAN
import io
import pyarrow.parquet as pq
import numpy as np

from PIL import Image, ImageCms
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import numpy as np


import numpy as np
from skimage import color

def rgb2lab(image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to the LAB color space and normalizes the values to the range [-1, 1].
    
    Args:
        image (np.ndarray): The input RGB image as a NumPy array with shape (H, W, 3).

    Returns:
        np.ndarray: The LAB image with values normalized to the range [-1, 1].
    """
    lab_image = color.rgb2lab(image)  # Convert RGB to LAB
    # Normalize L, a, b to the range [-1, 1]
    lab_image[..., 0] = (lab_image[..., 0] / 50.0) - 1.0  # Normalize L: [0, 100] -> [-1, 1]
    lab_image[..., 1:] = lab_image[..., 1:] / 128.0       # Normalize a, b: [-128, 127] -> [-1, 1]
    return lab_image

def lab2rgb(image: np.ndarray) -> np.ndarray:
    """
    Converts a normalized LAB image back to the RGB color space.
    
    Args:
        image (np.ndarray): The normalized LAB image as a NumPy array with shape (H, W, 3),
                            with values in the range [-1, 1].

    Returns:
        np.ndarray: The RGB image.
    """
    # Denormalize L, a, b to their original range
    lab_image = np.empty_like(image)
    lab_image[..., 0] = (image[..., 0] + 1.0) * 50.0   # Denormalize L: [-1, 1] -> [0, 100]
    lab_image[..., 1:] = image[..., 1:] * 128.0        # Denormalize a, b: [-1, 1] -> [-128, 127]
    rgb_image = color.lab2rgb(lab_image)  # Convert LAB back to RGB
    return rgb_image


class GAN_Dataset(Dataset):
    def __init__(self, data_path, split, img_size=256):
        self.img_size = img_size
        self.parquet_dataset = pq.ParquetFile(data_path)

        if split == "train":
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size), Image.BICUBIC),
                    # transforms.RandomHorizontalFlip(),
                ]
            )
        elif split == "val":
            self.transforms = transforms.Compose(
                [
                    # transforms.ToTensor(),
                    transforms.Resize((img_size, img_size), Image.BICUBIC),
                ]
            )

    def __getitem__(self, idx):
        row_group_size = (
            self.parquet_dataset.metadata.num_rows
            // self.parquet_dataset.metadata.num_row_groups
        )

        # Read a row group
        table = self.parquet_dataset.read_row_group(
            idx // row_group_size, columns=["image_bin"]
        )

        # Extract the image data
        image_data = table["image_bin"][idx % row_group_size - 1].as_py()

        # Convert binary data to image
        img = np.array(Image.open(io.BytesIO(image_data)).convert("RGB"))

        img = rgb2lab(img).astype(np.float32)
        img = self.transforms(img)
        
        # L = img[:, :, 0] / 255 - 1
        # ab = img[:, :, [1, 2]] / 255 - 1

        return img

    def __len__(self):
        return self.parquet_dataset.num_row_groups
