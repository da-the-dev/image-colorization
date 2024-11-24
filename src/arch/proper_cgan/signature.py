__all__ = ["signature"]

import torch
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec


signature = ModelSignature(
    Schema(TensorSpec(torch.float32, (-1, 1, 256, 256), "Input")),
    Schema(TensorSpec(torch.float32, (-1, 2, 256, 256), "Output")),
)
