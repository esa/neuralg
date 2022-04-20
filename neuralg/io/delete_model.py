from loguru import logger
import os


def delete_model(model_name):
    """Clears module from requested model state dict
    Args:
        model_name (str): Name of the custom model to delete
    """
    file_path = "../custom_models/" + str(model_name) + ".pt"
    if os.path.exists(file_path):
        logger.info(f"Deleting {model_name} from module")
        os.remove(file_path)
    else:
        logger.info(f"Path for {model_name} not found")
