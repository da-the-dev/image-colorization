__all__ = ["signature"]

import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec, DataType


signature = ModelSignature(
    Schema([TensorSpec(np.dtype(np.float32), (-1, 1, 256, 256), "Input")]),
    Schema([TensorSpec(np.dtype(np.float32), (-1, 2, 256, 256), "Output")]),
)
