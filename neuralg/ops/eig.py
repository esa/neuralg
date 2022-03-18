import numpy as np
import torch
from loguru import logger
import neuralg

# Potentially, we should perhaps call this eigvals, since it only computes eigenvalues
# Also note that is is only trained on symmetric matrices and will only output real eigenvalues.
#
def eig(A):
    """Approximates (real) eigenvalues of a square matrix. 
    Supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions. 
    Supports input of float and double dtypes.

    Args:
        A (tensor): Tensor of shape [*,d,d] where * can be zero or more batch dimensions.

    Returns:
        tensor: Containing the real-valued eigenvalue approximations to A. If A is a n-dimensional, resulting output is n-1 dimensional with the same batch dimension.
    """
    try:
        _validate_input(A)
    except ValueError:
        return None
    # if type(A) == np.ndarray:
    #     logger.info("Creating a tensor from input numpy ndarray ")
    #     A = torch.from_numpy(A)
    #     print(A.dtype)
    # Load the right model via model handler
    matrix_size = A.shape[-1]
    model = neuralg.neuralg_ModelHandler.get_model("eigval", matrix_size)
    # Evaluate model on input
    out = _predict(model, A)
    return out


def _predict(model, A):
    """Model prediction that allows keeping input dimension

    Args:
        model (torch.nn): Model to perform eigenvalue prediction
        A (tensor): Tensor of shape [*,d,d] where * can be zero or more batch dimension

    Returns:
        tensor: Containing eigenvalue approximations, of shape [*,d]
    """
    input_shape = A.shape
    if len(input_shape) == 2:
        # Add dummy dimension
        A = A[None, :]
        assert len(A.shape) == 3
        # Returns a one dimensional tensor
        out = model(A).squeeze()

    elif len(input_shape) == 3:
        A = A[:, None, :]
        out = model(A).squeeze(1)
    else:
        out = model(A)

    return out


def _validate_input(input):
    """ Checks that the input has correct shape. If not
    Args:
        input (tensor): Batch to be validated for eigenvalue approximation

    Raises:
        ValueError: If matrices are not quadratic
        ValueError: If input is not at least two dimensional
    """
    if len(input.shape) >= 2:
        None
    else:
        raise ValueError("Input must be at least two dimensional")

    if input.shape[-2] == input.shape[-1]:
        None
    else:
        raise ValueError("Matrices must be quadratic")
