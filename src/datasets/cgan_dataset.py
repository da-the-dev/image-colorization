# Здесь будет класс, наследуемый от Datasets для работы с GAN
import torch
import torchvision

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from skimage.color import rgb2lab, lab2rgb


class GAN_Dataset(Dataset):
    def __init__(self, data_path, split, img_size=256):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.RandomHorizontalFlip(), 
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((img_size, img_size),  Image.BICUBIC)
        
        self.data_path = data_path
        self.split = split
        self.img_size = img_size
    
    def __getitem__(self, idx):
        img = Image.open(self.data_path[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_to_lab = rgb2lab(img).astype("float32") 
        img_to_lab = transforms.ToTensor()(img_to_lab)
        L = img_to_lab[[0], ...] / 50. - 1. 
        ab = img_to_lab[[1, 2], ...] / 110. 
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.data_path)