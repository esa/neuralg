from multiprocessing.sharedctypes import Value
import torch
from ...utils.constants import (
    NEURALG_MIN_SYM_MATRIX_SIZE,
    NEURALG_MAX_SYM_MATRIX_SIZE,
    NEURALG_MAX_REAL_MATRIX_SIZE,
    NEURALG_MIN_REAL_MATRIX_SIZE,
    NEURALG_MAX_COMPLEX_MATRIX_SIZE,
    NEURALG_MIN_COMPLEX_MATRIX_SIZE,
    NEURALG_MAX_SVD_MATRIX_SIZE,
    NEURALG_MIN_SVD_MATRIX_SIZE,
    NEURALG_SUPPORTED_OPERATIONS,
)


def validate_input(input, operation, symmetric=None, real=None):

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


def _validate_eig_input(input, symmetric, real):
    """ Checks that the eig operation is supported for the passed matrix size.
    Args:
        input (tensor): Batch to be validated for eigenvalue approximation
        symmetric (bool): Specifying if matrix is symmetric.
        real (bool): Specyfing if output eigenvalues are enforced real

    Raises:
        ValueError: If matrices are not in supported size span, as defined in global constants
    """

    if symmetric:
        min_size, max_size = NEURALG_MIN_SYM_MATRIX_SIZE, NEURALG_MAX_SYM_MATRIX_SIZE
    elif real:
        min_size, max_size = NEURALG_MIN_REAL_MATRIX_SIZE, NEURALG_MAX_REAL_MATRIX_SIZE
    else:
        min_size, max_size = (
            NEURALG_MIN_COMPLEX_MATRIX_SIZE,
            NEURALG_MAX_COMPLEX_MATRIX_SIZE,
        )
    _validate_support(input, min_size, max_size)


def _validate_svd_input(input):
    """Checks that the svd operation is supported for the passed matrix size.

    Args:
        input (tensor): Batch to be validated for singular value approximation

    Raises:
        ValueError: If matrices are not in supported size span, as defined in global constants
    """
    _validate_support(input, NEURALG_MIN_SVD_MATRIX_SIZE, NEURALG_MAX_SVD_MATRIX_SIZE)


def _validate_support(input, min_size, max_size):
    if input.shape[-1] < min_size or input.shape[-1] > max_size:
        raise ValueError(
            "Matrix dimension for requested operation must be between {} and {}, but had dimension"
            + str(input.shape[-1]).format(min_size, max_size)
        )


def _general_validation(input):
    """ Checks that the input has correct shape.
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
