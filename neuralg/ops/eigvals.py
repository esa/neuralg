from .utils.predict import predict
from .. import neuralg_ModelHandler
from .. import neuralg_SafeMode
from .utils.validate_input import validate_input
from ..utils.constants import NEURALG_MATRIX_SIZES as MATRIX_SIZES


# Potentially, we should perhaps call this eigvals, since it only computes eigenvalues
def eigvals(
    A, symmetric=False, real=False, custom_model_name=None, custom_model_class=False
):
    """Approximates eigenvalues of a real valued square matrix.
     Supports batches of matrices, and if A is a batch of matrices then the output has the same batch dimensions.
     Supports input of float and double dtypes.
     Args:
        A (tensor): Tensor of shape [*,d,d] where * can be zero or more batch dimensions.
        symmetric (bool, optional): Specifying if matrix is symmetric, will load specialized models. Defaults to False.
        real (bool,optional): Specyfing if output eigenvalues should be enforced real, will load specialized trained models. Defaults to False.
        custom_model_name (str, optional): If specified and such model exists, the corresponding model will be used for inference. Defaults to None.
        custom_model_class (bool, optional): Specifies if custom model belongs to a custom class, will affect loading procedure. Defaults to False.
    Returns:
         tensor: Containing the real-valued eigenvalue approximations to A. If A is a n-dimensional, resulting output is n-1 dimensional with the same batch dimension.
    """
    if neuralg_SafeMode.mode:
        validate_input(A, operation="eig", symmetric=symmetric, real=real)

    matrix_size = A.shape[-1]
    # Load the right model via model handler
    if symmetric:
        try:
            model = neuralg_ModelHandler.get_model(
                "eigval", matrix_size, custom_model_name, custom_model_class
            )
        except ValueError:
            raise NotImplementedError(
                "Matrix dimension for symmetric eigval operation must be between {} and {}, but had dimension: ".format(
                    MATRIX_SIZES.eig.sym.lower_bound, MATRIX_SIZES.eig.sym.upper_bound
                )
                + str(A.shape[-1])
            )
    elif real:
        try:
            model = neuralg_ModelHandler.get_model(
                "r_eigval", matrix_size, custom_model_name, custom_model_class
            )
        except ValueError:
            raise NotImplementedError(
                "Matrix dimension for real eigval operation must be between {} and {}, but had dimension: ".format(
                    MATRIX_SIZES.eig.real.lower_bound, MATRIX_SIZES.eig.real.upper_bound
                )
                + str(A.shape[-1])
            )

    else:
        try:
            model = neuralg_ModelHandler.get_model(
                "c_eigval", matrix_size, custom_model_name, custom_model_class
            )
        except ValueError:
            raise NotImplementedError(
                "Matrix dimension for complex eigval operation must be between {} and {}, but had dimension: ".format(
                    MATRIX_SIZES.eig.complex.lower_bound,
                    MATRIX_SIZES.eig.complex.upper_bound,
                )
                + str(A.shape[-1])
            )

    out = predict(model, A)  # Evaluate model on input
    return out
