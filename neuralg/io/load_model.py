import torch
import os
from dotmap import DotMap
from neuralg.models.nerf import EigNERF
from neuralg.utils.constants import NEURALG_MIN_MATRIX_SIZE, NEURALG_MAX_MATRIX_SIZE


def load_model(model_name):
    """ Load a requested model, if available in the module. 

    Args:
        model_name (str): Name of requested model

    Returns:
        torch.nn : Requested model. Raises assertion if the model name not known. 
    """

    available_models = DotMap()
    # Added some trained nerf models for first minimal module
    for d in range(NEURALG_MIN_MATRIX_SIZE, NEURALG_MAX_MATRIX_SIZE + 1):
        state_dict_path = os.path.join(
            os.path.dirname(__file__), "../models/saved_models/eigval{}.pt".format(d),
        )
        available_models["eigval{}".format(d)] = [state_dict_path, "nerf", d]

    assert model_name in available_models, "Model not available"

    model_path, model_type, matrix_size = (x for x in available_models[model_name])

    if model_type == "nerf":
        assert type(matrix_size) == int
        assert matrix_size > 0
        model = EigNERF(matrix_size, in_features=matrix_size ** 2)
    else:
        raise NotImplementedError(
            "Unknown model type.  Available are: " + available_models.keys()
        )

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    return model
