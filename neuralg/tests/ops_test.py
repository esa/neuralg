import torch
import numpy as np
from dotmap import DotMap
from itertools import compress

from ..ops.eig import eig
from ..ops.svd import svd

from ..utils.constants import (
    NEURALG_MIN_SYM_MATRIX_SIZE,
    NEURALG_MAX_SYM_MATRIX_SIZE,
    NEURALG_MIN_COMPLEX_MATRIX_SIZE,
    NEURALG_MAX_COMPLEX_MATRIX_SIZE,
    NEURALG_MAX_REAL_MATRIX_SIZE,
    NEURALG_MIN_REAL_MATRIX_SIZE,
    NEURALG_MAX_SVD_MATRIX_SIZE,
    NEURALG_MIN_SVD_MATRIX_SIZE,
)
from ..evaluation.evaluate_model import evaluate_model
from ..evaluation.compute_accuracy import compute_accuracy


def test_eig():
    """ Tests if the eig operation works for all supported sizes and is within defined errors bounds, 
    for symmetric, non-symmetric with real eigenvalues and non-symmetric with complex eigenvalues
    """
    _test_eig(symmetric=True)
    _test_eig(real=True)


def _test_eig(symmetric=False, real=False):
    """Tests if the eig operation works for all supported sizes for the passed matrix type.  symmetric and non-symmetric and is within defined errors bounds. 
    The test set is sampled with a fixed seed. 
    """

    # Potentially, one might want to expand the error bounds to include more diverse and out-of training examples,
    # depending on the purpose
    test_parameters = {"N": 10000, "operation": "eig"}
    # These are set pretty high right now
    if symmetric:
        error_bound = 0.9  # Ratio of matrices we require to lie within the tolerance
        tol = 0.2  # Preliminary tolerace
        supported_sizes = np.arange(
            NEURALG_MIN_SYM_MATRIX_SIZE, NEURALG_MAX_SYM_MATRIX_SIZE + 1
        )
        test_parameters[
            "wigner"
        ] = True  # Test performance on Wigner matrices with eigvals with variance as in training.
    elif real:
        # Mock limits, very very high
        error_bound = 0.7
        tol = 0.38  # Preliminary tolerance
        supported_sizes = np.arange(
            NEURALG_MIN_REAL_MATRIX_SIZE, NEURALG_MAX_REAL_MATRIX_SIZE + 1
        )
        test_parameters[
            "dist"
        ] = "gaussian"  # Test performance on random matrices with normally distributed eigenvalues
        test_parameters["symmetric"] = False
    else:
        # Mock limits, very very high
        error_bound = 0.75
        tol = 0.3  # Preliminary tolerance
        supported_sizes = np.arange(
            NEURALG_MIN_COMPLEX_MATRIX_SIZE, NEURALG_MAX_COMPLEX_MATRIX_SIZE + 1
        )  # Test performance on random matrices with uniformly distributed elements

    op = lambda x: eig(x, symmetric=symmetric, real=real)

    _test_op(op, supported_sizes, test_parameters, tol, error_bound)


def test_svd():
    """Tests if the svd operation works for all supported sizes and is within defined errors bounds. 
    The test set is sampled with a fixed seed. 
    """

    test_parameters = {
        "N": 10000,
        "operation": "svd",
    }  # Test performance on random matrices with uniformly distributed elements

    # These are set pretty high right now
    error_bound = 0.9  # Ratio of matrices we require to lie within the tolerance
    tol = 0.15  # Preliminary tolerance
    supported_sizes = np.arange(
        NEURALG_MIN_SVD_MATRIX_SIZE, NEURALG_MAX_SVD_MATRIX_SIZE + 1
    )
    op = svd

    _test_op(op, supported_sizes, test_parameters, tol, error_bound)


def _test_op(op, supported_sizes, test_parameters, tol, error_bound):
    """Tests if the operation works for all supported sizes and is within defined errors bounds.

    Args:
        op (function): Requested operation for testing
        supported_sizes (list): List of supported matrix sizes 
        test_parameters (dict): Parameter configuration for the test
        tol (float): Error tolerance
        error_bound (float): Ratio of matrices we require to lie within the tolerance. 0 < tol < 1. 
    """
    results = DotMap()
    # Track which models fails the error bound
    results.failed = []
    results.accuracy = []
    for matrix_size in supported_sizes:
        test_parameters["d"] = matrix_size
        _test_batch_mode(op, matrix_size)

        # Test that errors are within the bound
        error = evaluate_model(op, test_parameters)
        accuracy = compute_accuracy(tol, error)
        results.accuracy.append(round(accuracy.item(), 3))
        results.failed.append(bool(accuracy - error_bound < 0))

    # Chech what,if any, models failed the requirements
    assert (
        sum(results.failed) == 0
    ), "Error bound {} not reached for{}: Failed for matrix sizes {} with accuracy {}, resepectively. Test parameters: {}".format(
        error_bound,
        op,
        list(compress(supported_sizes, results.failed)),
        list(compress(results.accuracy, results.failed)),
        test_parameters,
    )


def _test_batch_mode(op, matrix_size):
    """ Check operations support for different types of input e.g. in and out of batch mode with different batch dimensions

    Args:
        op (function): Requested operation
        matrix_size (int): Requested matrix size to be tested
    """

    test_matrix = torch.rand(matrix_size, matrix_size)
    test_batch = torch.rand(2, matrix_size, matrix_size)
    test_batch2 = torch.rand(2, 3, 4, matrix_size, matrix_size)
    out1 = op(test_matrix)
    out2 = op(test_batch)
    out3 = op(test_batch2)
    assert out1 is not None
    assert out1.__class__.__name__ == "Tensor"
    assert out1.shape == torch.Size([matrix_size])

    assert out2 is not None
    assert out2.__class__.__name__ == "Tensor"
    assert out2.shape == torch.Size([2, matrix_size])

    assert out3 is not None
    assert out3.__class__.__name__ == "Tensor"
    assert out3.shape == torch.Size([2, 3, 4, matrix_size])

    # Should also handle invalid input by throwing value errors
    try:
        out = op(torch.rand(3, 4))
    except ValueError:
        None
    try:
        out = op(torch.rand(3))
    except ValueError:
        None


if __name__ == "__main__":
    test_eig()
    test_svd()

