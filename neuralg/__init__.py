import os
from loguru import logger

__version__ = "0.0.10"  # TestPyPi index at the moment

from .utils.set_log_level import set_log_level

# Set default log level
set_log_level("WARNING")

from .io.download_models import download_models

# Check if directory of models exists
if not os.path.isdir(os.path.dirname(__file__) + "/models/saved_models"):
    # If not, they are dowonloaded from the latest release
    download_models()


# Set global safe mode variable to false default
neuralg_SAFEMODE = False

# Initialize global model handler
from .utils.ModelHandler import ModelHandler

neuralg_ModelHandler = ModelHandler()

# Add exposed features here
from .ops.eigvals import eigvals
from .ops.svd import svd

from .utils.set_precision import set_precision
from .utils.clear_loaded_models import clear_loaded_models
from .utils.set_up_torch import set_up_torch
from .utils.set_safe_mode import set_safe_mode
from .io.get_model import get_model
from .io.save_model import save_model
from .training.train_model import train_model


# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"

logger.info(f"Initialized neuralg for {os.environ['TORCH_DEVICE']}")

# Set default precision to float32 and uses CUDA if initialized
set_precision()

# Define exposed functions
__all__ = [
    "eigvals",
    "clear_loaded_models",
    "get_model",
    "save_model",
    "set_precision",
    "set_log_level",
    "set_up_torch",
    "set_safe_mode",
    "svd",
    "train_model",
]
