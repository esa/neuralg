import shutil
from loguru import logger
from ..utils.set_log_level import set_log_level


def clear_all_custom_model():
    """Deletes the custom models directory and all files contained in it"""
    set_log_level("WARNING")
    path = "../custom_models/"
    shutil.rmtree(path, ignore_errors=False, onerror=None)
    logger.warning("Removing all custom models from directory")
