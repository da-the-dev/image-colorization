# Здесь будет класс, наследуемый от Datasets для работы с GAN
import io
import pyarrow.parquet as pq
import numpy as np

from PIL import Image, ImageCms
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def rgb2lab(rgb_image: Image) -> Image:
    """Covert PIL RGB image to LAB.

    Args:
        rgb_image (Image): PIL RGB image.

    Returns:
        Image: Same image in LAB colorspace.
    """

    # Create profiles for each colorspace
    rgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")

    # Create a transform object
    rgb_to_lab_transform = ImageCms.buildTransform(
        inputProfile=rgb_profile,
        outputProfile=lab_profile,
        inMode="RGB",
        outMode="LAB",
    )

    # Apply the transform to convert RGB to LAB
    return ImageCms.applyTransform(rgb_image, rgb_to_lab_transform)


class GAN_Dataset(Dataset):
    def __init__(self, data_path, split, img_size=256):
        self.img_size = img_size
        self.parquet_dataset = pq.ParquetFile(data_path)

        if split == "train":
            self.transforms = transforms.Compose(
                [
                    # transforms.ToTensor(),
                    transforms.Resize((img_size, img_size), Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
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
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        img = rgb2lab(img)
        img = self.transforms(img)
        img = np.array(img).astype("float32")
        img = img.transpose((2, 1, 0))

        return img

    def __len__(self):
        return self.parquet_dataset.num_row_groups
