from loguru import logger
from neuralg.io.load_model import load_model
from neuralg.models.nerf import EigNERF, CEigNERF
from neuralg.utils.constants import NEURALG_SUPPORTED_OPERATIONS


def get_model(operation, matrix_dimension, symmetric=False, real=False, new=False):
    """ Loads a requested model from the module given a target operation and matrix size (and optional matrix properties for eig operation). 
        Also enable instantiating a new model, upon request.

    Args:
        operation (str): Requested linear algebra operation, e.g. eig or svd
        matrix_dimension (int): Matrix size with which the model should support
        symmetric (bool, optional): Specifies if model will predict on symmetric matrices. Only applicable to eig operation. Defaults to False.
        real (bool, optional): Specifies if model will predict on matrices with only real eigenvalues. Only applicable to eig operation.  Defaults to False.
        new (bool, optional): If true, a new model without previous training will be instantiated. Defaults to False.

    Returns:
        torch.nn: Model 
    """
    assert (
        str(operation) in NEURALG_SUPPORTED_OPERATIONS
    ), f" Requested operation {operation} unavailable, must be in {NEURALG_SUPPORTED_OPERATIONS}"

    if new:  # Instantiate a new model
        logger.info(
            f"Instantiating a model without previous training for {operation} operation"
        )
        if str(operation) == "eig" and not symmetric and not real:
            logger.warning(f"Outputs of the requested model are ComplexFloat dtype")
            model = CEigNERF(matrix_dimension, matrix_dimension ** 2)
        else:
            model = EigNERF(matrix_dimension, matrix_dimension ** 2)
    else:  # Load model saved in module
        logger.info(f"Loading a pre-trained model for {operation} operation")
        if str(operation) == "eig":
            if symmetric:
                model_name = f"eigval{matrix_dimension}"
            elif real:
                model_name = f"r_eigval{matrix_dimension}"
            else:
                model_name = f"c_eigval{matrix_dimension}"
        elif str(operation) == "svd":
            model_name = f"svd{matrix_dimension}"

        model = load_model(model_name)
    model.train()  # Activate training mode

    return model

