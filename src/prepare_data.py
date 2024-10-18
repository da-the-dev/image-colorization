# Здесь будут функции для подготовки датасета и даталоудера
import os
import glob
import sys
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
import numpy as np


ImageFile.LOAD_TRUNCATED_IMAGES = True

#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

def filter_corrupted_images(file_paths):
    valid_images = []
    corrupted_images = []

    for path in file_paths:
        try:
            img = Image.open(path)
            img.verify()  # Verify if the image is valid
            valid_images.append(path)  # Add valid images to the list
        except (IOError, SyntaxError) as e:
            corrupted_images.append(path)  # Add corrupted images to the list
            #print(f"Corrupted image removed: {path} - {e}")

    print(f"Total corrupted images removed: {len(corrupted_images)}")
    return valid_images


# нужно накинуть гидру и как параметры все параметры функции
def get_data(data_path='data/coco2017/train2017', num=10, persent_for_val = 0.2):

    #берем все названия картинок
    img_paths = glob.glob(data_path + "/*.jpg") 

    # сколько всего картинок пойдет в трейн
    num_train_samples = num - int(num * persent_for_val)

    # берем первые num картинок
    img_paths_sub = np.array(img_paths[:num])

    # генерируем рандомные индексы, чтобы брать рандомные картинки
    rand_ids = np.random.permutation(num)

    # разделим на 

    # train
    train_ids = rand_ids[:num_train_samples] 
    train_paths = img_paths_sub[train_ids]

    # val
    val_ids = rand_ids[num_train_samples:] 
    val_paths = img_paths_sub[val_ids]

    print("Train size: ", len(train_paths))
    print("Val size: ", len(val_paths))

    train_paths = filter_corrupted_images(train_paths)
    val_paths = filter_corrupted_images(val_paths)


    print("Train size after filtering: ", len(train_paths))
    print("Val size after filtering: ", len(val_paths))


    return train_paths, val_paths




if __name__ == "__main__":
    train_paths, val_paths = get_data()

    print(train_paths)
    print(val_paths)
