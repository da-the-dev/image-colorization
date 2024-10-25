import torch


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

