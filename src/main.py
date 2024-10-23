# Здесь будет прописана основная работа программы
from prepare_data import get_data
from transform_data import transform_GAN_data
from model import GNet, GAN_Model
import torch

def run_default_GAN():
    train_paths, val_paths = get_data(num=1000)

    train_dl = transform_GAN_data(train_paths, split='train')
    val_dl = transform_GAN_data(val_paths, split='val')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_G = GNet(device, optimizer="Adam", body="resnet34")

    model_G.pretrain(train_dl, epochs=5)

    model_G.save_model(path="../models/GAN/model_G.pt")

    GAN_model = GAN_Model(model_G.G_net, lr_G=0.0004, lr_D=0.0004, beta1=0.5, beta2=0.999, lamda=100.)
    GAN_model.train_model(train_dl, epochs=5)

    GAN_model.save_model(path="../models/GAN/model_GAN.pt")





# сюда можно докинуть hydra и юзать как параметр название архитектуры
def main(archetecture):
    if archetecture == "GAN":
        run_default_GAN()


main("GAN")