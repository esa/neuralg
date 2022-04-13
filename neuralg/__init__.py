import os
from loguru import logger

# Add exposed features here
from .ops.eigvals import eigvals
from .ops.svd import svd
from .utils.ModelHandler import ModelHandler
from .utils.set_log_level import set_log_level
from .utils.set_precision import set_precision
from .utils.clear_loaded_models import clear_loaded_models
from .utils.set_up_torch import set_up_torch

set_log_level("INFO")

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"

logger.info(f"Initialized neuralg for {os.environ['TORCH_DEVICE']}")

# Set default precision to float32 and uses CUDA if initialized
set_precision()

# Initialize global model handler
neuralg_ModelHandler = ModelHandler()


__all__ = [
    "eigvals",
    "clear_loaded_models",
    "set_precision",
    "set_log_level",
    "set_up_torch",
    "svd",
]
