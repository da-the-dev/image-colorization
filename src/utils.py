import torch


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def instance(name: str):
    import importlib

    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]

    return getattr(importlib.import_module(module_name), class_name)
