from ast import Assert
from neuralg.io.load_model import load_model


class ModelHandler:
    def __init__(self):
        self.loaded_models = {}

    def ship_model(self, op, matrix_size):
        requested_model_name = self.get_model_name(op, matrix_size)
        if requested_model_name in self.loaded_models:
            return self.loaded_models[requested_model_name]
        else:
            return self.first_load(requested_model_name)

    def first_load(self, model_name):
        try:
            loaded_model = load_model(model_name)
            # Track all loaded models
            self.loaded_models[model_name] = loaded_model
            return loaded_model
        except AssertionError:
            raise ValueError("Requested operation is not available")

    def get_model_name(self, op, matrix_size, other_options=None):
        if other_options is None:
            model_name = "{}{}".format(
                op, matrix_size
            )  # This might not be a genius way to do it
        else:
            model_name = None
        return model_name

