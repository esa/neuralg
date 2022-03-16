import os
import torch
import sys
from loguru import logger

# Add exposed features here
from .ops.eig import eig
from .utils.load_default_cfg import load_default_cfg
from .scripts.train_models import train_models
from .utils.ModelHandler import ModelHandler
from .utils.set_log_level import set_log_level
from .utils.print_cfg import print_cfg
from .plots.plot_loss_logs import plot_loss_logs
from .plots.plot_eigval_results import plot_eigval_results
from .training.save_run import save_run
from .training.evaluate import evaluate

set_log_level("INFO")

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"

logger.info(f"Initialized neuralg for {os.environ['TORCH_DEVICE']}")

# Set precision (and potentially GPU)
torch.set_default_tensor_type(torch.DoubleTensor)
logger.info("Using double precision")

# Initialize global model handler
neuralg_ModelHandler = ModelHandler()

<<<<<<< HEAD
#
__all__ = ["eig"]
=======
# Potentially in the final module not all of these should be imported,
# but I have it right now for the notebook.
__all__ = [
    "eig",
    "evaluate",
    "load_default_cfg",
    "load_model",
    "plot_loss_logs",
    "plot_eigval_results",
    "print_cfg",
    "save_run",
    "train_models",
]
>>>>>>> d9fad2a9f40f6342f4d5b8c00c25ace2311a4822
