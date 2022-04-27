from loguru import logger
from .. import neuralg_SAFEMODE


def set_safe_mode(safe_mode=False):
    global neuralg_SAFEMODE
    """ Enable running checks on inputs and outputs (e.g. eigval > 1e16 , input NaN etc.)

    Args:
        safe_mode (bool, optional): If true, checks are run on input and output. Defaults to False.
    """
    assert type(safe_mode) is bool, "Input must be bool"

    if safe_mode == neuralg_SAFEMODE:
        logger.info(f"Safe mode already set to {safe_mode}")
    else:
        neuralg_SAFEMODE = safe_mode
        if safe_mode:
            logger.info("Activating safe mode. Input and outputs will be monitored")
        elif not safe_mode:
            logger.info("De-activating safe mode")
