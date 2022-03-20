import os
from loguru import logger

# Add exposed features here
from .ops.eig import eig
from .utils.ModelHandler import ModelHandler
from .utils.set_log_level import set_log_level
from .utils.set_precision import set_precision
from .utils.clear_loaded_models import clear_loaded_models

set_log_level("INFO")

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"

logger.info(f"Initialized neuralg for {os.environ['TORCH_DEVICE']}")

# Set default precision
set_precision()

# Initialize global model handler
neuralg_ModelHandler = ModelHandler()


__all__ = ["eig", "set_precision", "set_log_level", "clear_loaded_models"]
