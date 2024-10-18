# Здесь будут прописаны функции для преобразования данных
#from prepare_data import get_data

from custom_datasets.gan_dataset import GAN_Dataset

from torch.utils.data import Dataset, DataLoader


# гидру сделать и поставить батчсайз
def transform_GAN_data(paths, split, batch_size=128):
    cur_dataset = GAN_Dataset(paths, split=split)
    
    print(f"{split} dataset length: ", len(cur_dataset))

    cur_dl = DataLoader(cur_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    print(f"{split} dataloader length: ", len(cur_dl))

    return cur_dl



if __name__ == "__main__":

    # train_paths, val_paths = get_data()
    # train_dl = transform_data(train_paths, split='train')
    # val_dl = transform_data(val_paths, split='val')


    # print(train_dl)
    # print(val_dl)
    pass