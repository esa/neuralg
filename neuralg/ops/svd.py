from .. import neuralg_ModelHandler
from .utils.validate_input import validate_input


def svd(A, custom_model_name=None):
    """Approximates  singular values of a square matrix.
     Supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions.
     Supports input of float and double dtypes.
     Args:
         A (tensor): Tensor of shape [*,d,d] where * can be zero or more batch dimensions.
         custom_model_name (str,optional): If specified, the custom model with passed name will be used in approximation. Defaults to None
    Returns:
         tensor: Containing the real-valued singular value approximations to A. If A is a n-dimensional, resulting output is n-1 dimensional with the same batch dimension.
    """

    validate_input(A, operation="svd")

    matrix_size = A.shape[-1]

    model = neuralg_ModelHandler.get_model(
        "svd", matrix_size
    )  # Load the right model via model handler
    out = model(A)  # Evaluate model on input
    return out
