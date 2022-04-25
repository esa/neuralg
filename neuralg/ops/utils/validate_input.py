import torch
from loguru import logger
from ...utils.constants import (
    NEURALG_MATRIX_SIZES as MATRIX_SIZES,
    NEURALG_SUPPORTED_OPERATIONS,
)


def validate_input(input, operation, symmetric=False, real=False):
    """Checks that the input is valid for the requested operation

    Args:
        input (tensor): Input to be validated
        operation (str): Operation requested for the input, e.g. eig or svd.
        symmetric (bool, optional): Only applies to eig operation. Defaults to None.
        real (bool, optional): Only applies to eig operation. Defaults to None.

    Raises:
        ValueError: If the operation is not supported for the passed input shape and type
    """

    _general_validation(input)

    if operation == "eig":
        _validate_eig_input(input, symmetric, real)
    elif operation == "svd":
        _validate_svd_input(input)
    else:
        raise ValueError(
            "Operation {} not supported, must be in {} ".format(
                operation, NEURALG_SUPPORTED_OPERATIONS
            )
        )

    _safety_check(input)


def _validate_eig_input(input, symmetric, real):
    """Checks that the eig operation is supported for the passed matrix size.
    Args:
        input (tensor): Batch to be validated for eigenvalue approximation
        symmetric (bool): Specifying if matrix is symmetric.
        real (bool): Specyfing if output eigenvalues are enforced real

    Raises:
        ValueError: If matrices are not in supported size span, as defined in global constants
    """
    if symmetric:
        min_size, max_size = (
            MATRIX_SIZES.eig.sym.lower_bound,
            MATRIX_SIZES.eig.sym.upper_bound,
        )
    elif real:
        min_size, max_size = (
            MATRIX_SIZES.eig.real.lower_bound,
            MATRIX_SIZES.eig.real.upper_bound,
        )
    else:
        min_size, max_size = (
            MATRIX_SIZES.eig.complex.lower_bound,
            MATRIX_SIZES.eig.complex.upper_bound,
        )
    _validate_support(input, min_size, max_size)


def _validate_svd_input(input):
    """Checks that the svd operation is supported for the passed matrix size.

    Args:
        input (tensor): Batch to be validated for singular value approximation

    Raises:
        ValueError: If matrices are not in supported size span, as defined in global constants
    """
    _validate_support(input, MATRIX_SIZES.svd.lower_bound, MATRIX_SIZES.svd.upper_bound)


def _validate_support(input, min_size, max_size):
    """Checks if the input matrix size is supported

    Args:
        input (tensor): Input to be validated
        min_size (int): Minimium supported matrix size
        max_size (int): Minimium supported matrix size

    Raises:
        ValueError: If matrix size is out of bounds
    """
    if input.shape[-1] < min_size or input.shape[-1] > max_size:
        raise ValueError(
            "Matrix dimension for requested operation must be between {} and {}, but had dimension: ".format(
                min_size, max_size
            )
            + str(input.shape[-1])
        )


def _general_validation(input):
    """Checks that the input has correct shape.
    Args:
        input (tensor): Batch to be validated

    Raises:
        ValueError: If input is not of torch.Tensor type
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


def _safety_check(input):
    """Run safety check on model inputs.

    Args:
        input (tensor): Input tensor to check

    Raises:
        ValueError: If NaN input

    Yields:
        Warning: If input elements exceed 1e16 in abolute value
    """
    max_lim = 1e16
    if torch.isnan(input).sum() != 0:
        raise ValueError("NaN input not supported")
    if input.abs().max() > max_lim:
        logger.warning(
            f"Input elements exceed {max_lim} in absolute value. Might yield unexpected output"
        )
