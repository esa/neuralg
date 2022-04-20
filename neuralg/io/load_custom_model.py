import torch
import os
from ..models.nerf import EigNERF, CEigNERF


def load_custom_model(op, matrix_size, model_name, custom_class=False):
    """Load a requested model, if available in the module.

    Args:
        op (str): Requested linear algebra operation
        matrix_size (int): Size of matrices operation is defined for.
        model_name (str): Name of requested model as saved in file.

    Returns:
        torch.nn : Requested model. Raises assertion if the model name not known.
    """

    model_path = os.path.realpath(
        "../custom_models/{}.pt".format(model_name),
    )
    if custom_class:
        model = torch.load(model_path)
    else:
        if (
            op == "c_eigvals"
        ):  # Currently this is the only operation requiring complex outputs.
            model = CEigNERF(matrix_size, in_features=matrix_size**2)
        else:
            model = EigNERF(matrix_size, in_features=matrix_size**2)

        # Load designated model state dict from path
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict=state_dict)

    model.eval()
    return model
