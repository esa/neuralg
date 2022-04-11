import torch
import neuralg
from ..utils.constants import NEURALG_MIN_MATRIX_SIZE, NEURALG_MAX_MATRIX_SIZE

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
    _validate_input(A)
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
    out = model(A)
    return out


def _validate_input(input):
    """ Checks that the input has correct shape.
    Args:
        input (tensor): Batch to be validated for eigenvalue approximation
    Raises:
        ValueError: If matrices are not quadratic
        ValueError: If input is not at least two dimensional
    """
    if not torch.is_tensor(input):
        raise ValueError("Only torch tensors are supported as input")

    if len(input.shape) < 2:
        raise ValueError(
            "Input must be at least two dimensional, but had shape" + str(input.shape)
        )
    if input.shape[-2] != input.shape[-1]:
        raise ValueError("Matrices must be quadratic but had shape" + str(input.shape))

    if (
        input.shape[-1] < NEURALG_MIN_MATRIX_SIZE
        or input.shape[-1] > NEURALG_MAX_MATRIX_SIZE
    ):
        raise ValueError(
            "Matrix dimension must be between {} and {}, but had dimension"
            + str(input.shape[-1]).format(
                NEURALG_MIN_MATRIX_SIZE, NEURALG_MAX_MATRIX_SIZE
            )
        )
