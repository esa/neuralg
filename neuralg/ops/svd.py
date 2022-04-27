from .. import neuralg_ModelHandler
from .. import neuralg_SAFEMODE
from .utils.validate_input import validate_input
from .utils.predict import predict


def svd(A, custom_model_name=None, custom_model_class=False):
    """Approximates  singular values of a square matrix.
     Supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions.
     Supports input of float and double dtypes.
     Args:
        A (tensor): Tensor of shape [*,d,d] where * can be zero or more batch dimensions.
        custom_model_name (str,optional): If specified, the custom model with passed name will be used in approximation. Defaults to None.
        custom_model_class (bool,optional): Specifies if custom model belongs to a custom class, will affect loading procedure. Defaults to False.
    Returns:
         tensor: Containing the real-valued singular value approximations to A. If A is a n-dimensional, resulting output is n-1 dimensional with the same batch dimension.
    """
    if neuralg_SAFEMODE:
        validate_input(A, operation="svd")

    matrix_size = A.shape[-1]

    model = neuralg_ModelHandler.get_model(
        "svd", matrix_size, custom_model_name, custom_model_class
    )  # Load the right model via model handler
    out = predict(model, A)  # Evaluate model on input
    return out
