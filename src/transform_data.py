from src.datasets.cgan_dataset import GAN_Dataset

from torch.utils.data import DataLoader


# гидру сделать и поставить батчсайз
def transform_GAN_data(paths, split, batch_size=128):
    cur_dataset = GAN_Dataset(paths, split=split)

    print(f"{split} dataset length: ", len(cur_dataset))

    cur_dl = DataLoader(
        cur_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    print(f"{split} dataloader length: ", len(cur_dl))

    return cur_dl
