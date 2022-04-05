import torch
import numpy as np
from dotmap import DotMap
from itertools import compress

from ..ops.eig import eig
from ..utils.constants import (
    NEURALG_MIN_SYM_MATRIX_SIZE,
    NEURALG_MAX_SYM_MATRIX_SIZE,
    NEURALG_MIN_COMPLEX_MATRIX_SIZE,
    NEURALG_MAX_COMPLEX_MATRIX_SIZE,
    NEURALG_MAX_REAL_MATRIX_SIZE,
    NEURALG_MIN_REAL_MATRIX_SIZE,
)
from ..evaluation.evaluate_model import evaluate_model
from ..evaluation.compute_accuracy import compute_accuracy


def test_eig(symmetric=False, real=False):
    """Tests if the eig operation works for all supported sizes, symmetric and non-symmetric and is within defined errors bounds. 
    The test set is sampled with a fixed seed. 
    """

    # Potentially, I think one will want to expand the error bounds to include more diverse and out-of training examples,
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
        tol = 0.3  # Preliminary tolerance
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
        )
        # Test performance on random matrices with uniformly distributed elements

        results = DotMap()
        # Track which models fails the error bound
        results.failed = []
        results.accuracy = []
        for matrix_size in supported_sizes:
            test_parameters["d"] = matrix_size
            # Make sure that it supports different types of input e.g. in and out of batch mode with different batch dimensions
            test_matrix = torch.rand(matrix_size, matrix_size)
            test_batch = torch.rand(2, matrix_size, matrix_size)
            test_batch2 = torch.rand(2, 3, 4, matrix_size, matrix_size)
            out1 = eig(test_matrix, symmetric)
            out2 = eig(test_batch, symmetric)
            out3 = eig(test_batch2, symmetric)
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
                out = eig(torch.rand(3, 4), symmetric)
            except ValueError:
                None
            try:
                out = eig(torch.rand(3), symmetric)
            except ValueError:
                None

            # Test that errors are within the bound
            error = evaluate_model(lambda x: eig(x, symmetric), test_parameters)
            accuracy = compute_accuracy(tol, error)
            results.accuracy.append(round(accuracy.item(), 3))
            results.failed.append(bool(accuracy - error_bound < 0))

        # Chech what,if any, models failed the requirements
        assert (
            sum(results.failed) == 0
        ), "Error bound not reached: Failed for matrix sizes {} with accuracy {}, resepectively. Symmetric = {}, Real = {}".format(
            list(compress(supported_sizes, results.failed)),
            list(compress(results.accuracy, results.failed)),
            str(symmetric),
            str(real),
        )


if __name__ == "__main__":
    test_eig(symmetric=True)
    test_eig(real=True)
    test_eig()

