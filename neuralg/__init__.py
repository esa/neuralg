import os
import torch
import sys
from loguru import logger

# TODO

# Add exposed features here
# I am having problem with this relative import, where I have to call neural.whatever,
# which is not really correct, right?
from neuralg.ops.eig import eig
from neuralg.utils.load_default_cfg import load_default_cfg
from neuralg.scripts.train_models import train_models
from neuralg.utils.ModelHandler import ModelHandler
from neuralg.utils.set_log_level import set_log_level
from neuralg.utils.print_cfg import print_cfg
from neuralg.plots.plot_loss_logs import plot_loss_logs
from neuralg.plots.plot_eigval_results import plot_eigval_results
from neuralg.training.save_run import save_run
from neuralg.training.evaluate import evaluate

set_log_level("INFO")

# Set main device by default to cpu if no other choice was made before
if "TORCH_DEVICE" not in os.environ:
    os.environ["TORCH_DEVICE"] = "cpu"

logger.info(f"Initialized neuralg for {os.environ['TORCH_DEVICE']}")

# Set precision (and potentially GPU)
# torch.set_default_tensor_type(torch.DoubleTensor)
# logger.info("Using double precision")

# Initialize global model handler?
neuralg_ModelHandler = ModelHandler()

# Potentially in the final module not all of these should be imported,
# but I have it right now for the notebook.
__all__ = [
    "eig",
    "evaluate",
    "loaf_default_cfg",
    "load_model",
    "plot_loss_logs",
    "plot_eigval_results",
    "print_cfg",
    "save_run",
    "train_models",
]
