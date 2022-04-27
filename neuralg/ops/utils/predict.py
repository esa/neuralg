from ... import neuralg_SAFEMODE
import torch
from loguru import logger


def predict(model, input):
    """Predicts output with model on passed input with potential safety checks on outputs

    Args:
        model (torch.nn): Model for prediction
        input (tensor): Input to predict on

    Returns:
        tensor : Model output given passed input
    """
    out = model(input)
    if neuralg_SAFEMODE:
        _check_output(out)
    return out


def _check_output(out):
    """Run safety check on model outputs. Logger warns if potential extreme or abnormal outputs.

    Args:
        out (tensor): Output tensor to check
    """
    if torch.isnan(out).sum() != 0:
        logger.warning(f"Output is NaN ")
    max_lim = 1e16
    if out.abs().max() > max_lim:
        logger.warning(f"Output elements exceed {max_lim} in modulus")
