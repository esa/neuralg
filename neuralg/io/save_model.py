from loguru import logger
import torch
from pathlib import Path


def save_model(model, model_name, custom_class=False):
    """Saves model state dict to .pt file.
    If it doesnt exist, creates a custom folder where model state dicts are saved.

    Args:
        model (torch.nn): Torch model requested to save
        model_name (str): Name of the custom model
    """

    # Create subfolder if it does not exist
    Path("../custom_models/").mkdir(parents=True, exist_ok=True)
    filename = "../custom_models/" + str(model_name) + ".pt"

    logger.info("Saving model to file: {}".format(filename))
    if custom_class:
        torch.save(model, filename)
    else:
        torch.save(model.state_dict(), filename)
