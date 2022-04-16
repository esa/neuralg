import torch
import os
from dotmap import DotMap
from ..models.nerf import EigNERF, CEigNERF
from ..utils.constants import NEURALG_MATRIX_SIZES as MATRIX_SIZES


def load_model(model_name):
    """ Load a requested model, if available in the module. 

    Args:
        model_name (str): Name of requested model

    Returns:
        torch.nn : Requested model. Raises assertion if the model name is not known. 
    """

    available_models = DotMap()
    # Models trained on symmetric matrices
    for d in range(
        MATRIX_SIZES.eig.sym.lower_bound, MATRIX_SIZES.eig.sym.upper_bound + 1
    ):
        state_dict_path = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "../models/saved_models/eigval{}.pt".format(d),
            )
        )
        available_models["eigval{}".format(d)] = [state_dict_path, "nerf", d]

    # Models trained on non-symmetric matrices with real eigenvalues
    for d in range(
        MATRIX_SIZES.eig.real.lower_bound, MATRIX_SIZES.eig.real.upper_bound + 1
    ):
        state_dict_path = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "../models/saved_models/r_eigval{}.pt".format(d),
            )
        )
        available_models["r_eigval{}".format(d)] = [state_dict_path, "nerf", d]

    # Models trained on non-symmetric matrices with potentially complex eigenvalues
    # Still very much in prototyping
    for d in range(
        MATRIX_SIZES.eig.complex.lower_bound, MATRIX_SIZES.eig.complex.upper_bound + 1
    ):
        state_dict_path = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "../models/saved_models/c_eigval{}.pt".format(d),
            )
        )
        available_models["c_eigval{}".format(d)] = [state_dict_path, "complex_nerf", d]

    # Models trained on non-symmetric matrices with potentially complex eigenvalues
    for d in range(MATRIX_SIZES.svd.lower_bound, MATRIX_SIZES.svd.upper_bound + 1):
        state_dict_path = os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                "../models/saved_models/svd/svd{}.pt".format(d),
            )
        )
        available_models["svd{}".format(d)] = [state_dict_path, "nerf", d]

    assert model_name in available_models, "Model not available, must be in {}".format(
        list(available_models.keys())
    )
    model_path, model_type, matrix_size = (x for x in available_models[model_name])

    if model_type == "nerf":
        assert type(matrix_size) == int
        assert matrix_size > 0
        model = EigNERF(matrix_size, in_features=matrix_size ** 2)
    elif model_type == "complex_nerf":
        assert type(matrix_size) == int
        assert matrix_size > 0
        model = CEigNERF(matrix_size, in_features=matrix_size ** 2)
    else:
        raise NotImplementedError(
            "Unknown model type.  Available are: " + available_models.keys()
        )

    # Load designated model state dict from path
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    return model
