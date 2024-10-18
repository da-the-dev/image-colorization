# Здесь будет прописана основная работа программы
from prepare_data import get_data
from transform_data import transform_GAN_data

def run_default_GAN():
    train_paths, val_paths = get_data(num=1000)

    train_dl = transform_GAN_data(train_paths, split='train')
    val_dl = transform_GAN_data(val_paths, split='val')



# сюда можно докинуть hydra и юзать как параметр название архитектуры
def main(archetecture):
    if archetecture == "GAN":
        run_default_GAN()


main("GAN")