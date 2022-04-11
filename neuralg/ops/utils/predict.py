import neuralg
from loguru import logger
from ...utils.set_log_level import set_log_level


def predict(model, input):
    """ Predicts output with model on passed input with potential safety checks on outputs

    Args:
        model (torch.nn): Model for prediction
        input (tensor): Input to predict on

    Returns:
        tensor : Model output given passed input
    """
    out = model(input)
    if neuralg.neuralg_SAFEMODE:
        _check_output(out)
    return out


def _check_output(out):
    """ Run safety check on model outputs. Logger warns if potential extreme or abnormal outputs. 

    Args:
        out (tensor): Output tensor to check 
    """
    set_log_level("WARNING")
    max_lim = 1e16
    if out.abs().max() > max_lim:
        logger.warning(f"Output elements exceeds {max_lim} in modulus")

