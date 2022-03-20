from copy import deepcopy
from neuralg.io.load_model import load_model
from loguru import logger


class ModelHandler:
    """Tracks and stores loaded models and provides them to the ops.
    
    When a model is requested for the first time it loads it. Supports clearing the loaded models.
    """

    def __init__(self):
        self.loaded_models = {}

    def get_model(self, op, matrix_size):
        """ Get the designated model for given operation and matrix size

        Args:
            op (str): The linear algebra operation the model should approximate
            matrix_size (int): Size of the matrices the model should approximate operation on

        Returns:
            torch.nn: Requested designated model
        """
        requested_model_name = self._get_model_name(op, matrix_size)
        if requested_model_name in self.loaded_models:
            return self.loaded_models[requested_model_name]
        else:
            return self.first_load(requested_model_name)

    def first_load(self, model_name):
        """ Loads a model for the first time

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

    def clear_loaded_models(self):
        """ Frees allocated memory from loaded models
        """
        logger.info("Clearing loaded models")
        loaded_models_copy = deepcopy(self.loaded_models)
        for model in loaded_models_copy:
            del self.loaded_models[model]
        del loaded_models_copy  # Not sure if needed
        None

