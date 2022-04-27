from copy import deepcopy
from ..io.load_model import load_model
from ..io.load_custom_model import load_custom_model
from loguru import logger


class ModelHandler:
    """Tracks and stores loaded models and provides them to the ops.

    When a model is requested for the first time it loads it. Supports clearing the loaded models.
    """

    def __init__(self):
        self.loaded_models = {}

    def get_model(
        self, op, matrix_size, custom_model_name=None, custom_model_class=False
    ):
        """Get the designated model for given operation and matrix size

        Args:
            op (str): The linear algebra operation the model should approximate
            matrix_size (int): Size of the matrices the model should approximate operation on
            custom_model_path (str,optional): If specified, the custom model with passed name will be used in approximation. Defaults to None
            custom_model_class (bool,optional): Specifies if custom model belongs to a custom class, will affect loading procedure. Defaults to False.

        Returns:
            torch.nn: Requested designated model
        """
        if custom_model_name is not None:
            return self._get_custom_model(
                op, matrix_size, custom_model_name, custom_model_class
            )
        else:
            requested_model_name = self._get_model_name(op, matrix_size)
        if requested_model_name in self.loaded_models:
            return self.loaded_models[requested_model_name]
        else:
            return self._first_load(requested_model_name)

    def _first_load(self, model_name):
        """Loads a model for the first time

        Args:
            model_name (str): Name of the requested model

        Returns:
            torch.nn : Requested model
        """
        loaded_model = load_model(model_name)

        # Track all loaded models
        self.loaded_models[model_name] = loaded_model
        return loaded_model

    def _get_model_name(self, op, matrix_size):
        """Formats the correct name of designated model for the specified operation and matrix size

        Args:
            op (str): The linear algebra operation the model should approximate
            matrix_size (int): Size of the matrices the model should approximate operation on
        Returns:
            str: Formatted name of model as named in saved_models
        """
        model_name = "{}{}".format(op, matrix_size)
        return model_name

    def _get_custom_model(self, op, matrix_size, model_name, custom_model_class):
        """If it exists, loads a custom model via path for the given operation and matrix size.

        Args:
            op (str):  The linear algebra operation the model should approximate
            matrix_size (int):  Size of the matrices the model should approximate operation on
            model_name (str): Name of the requested model as saved in file.

        Returns:
            torch.nn: Requested custom model.
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        else:
            loaded_model = load_custom_model(
                op, matrix_size, model_name, custom_model_class
            )
            # Track all loaded models
            self.loaded_models[model_name] = loaded_model
        return loaded_model

    def clear_loaded_models(self):
        """Frees allocated memory from loaded models"""
        logger.info("Clearing loaded models")
        loaded_models_copy = deepcopy(list(self.loaded_models.keys()))
        for model in loaded_models_copy:
            del self.loaded_models[model]
        del loaded_models_copy
