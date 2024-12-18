import io
import numpy as np
import pyarrow.parquet as pq
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

from src.arch.proper_cgan.utils import rgb2lab_normalize


def make_transforms(img_size):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
        ]
    )


class GAN_Dataset(Dataset):
    def __init__(self, data_path, img_size=256):
        self.img_size = img_size
        self.parquet_dataset = pq.ParquetFile(data_path)

        transforms = self.make_transforms(img_size)

        self.transforms = transforms

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

        img = rgb2lab_normalize(img).astype(np.float32)
        img = self.transforms(img)

        return img

    def __len__(self):
        return self.parquet_dataset.num_row_groups
