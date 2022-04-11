import torch
import numpy as np
from dotmap import DotMap
from itertools import compress

from ..ops.eig import eig
from ..utils.constants import NEURALG_MIN_MATRIX_SIZE, NEURALG_MAX_MATRIX_SIZE
from ..evaluation.evaluate_model import evaluate_model
from ..evaluation.compute_accuracy import compute_accuracy


def test_eig():
    """Tests if the eig operation works for all supported sizes and is within defined errors bounds. 
    The test set is sampled with a fixed seed. 
    """
    supported_sizes = np.arange(NEURALG_MIN_MATRIX_SIZE, NEURALG_MAX_MATRIX_SIZE + 1)

    # Initially, we only test performance on Wigner matrices with eigvals with variance as in training.
    # Potentially, I think one will want to expand the error bounds to include more diverse and out-of training examples,
    # depending on the purpose
    test_parameters = {"N": 10000, "operation": "eig", "wigner": True}

    # These are set pretty high right now
    tol = 0.16  # Preliminary tolerace
    error_bound = 0.95  # Percentage of matrices we require to lie within the tolerance
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
        out1 = eig(test_matrix)
        out2 = eig(test_batch)
        out3 = eig(test_batch2)
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
            out = eig(torch.rand(3, 4))
        except ValueError:
            None
        try:
            out = eig(torch.rand(3))
        except ValueError:
            None

        # Test that errors are within the bound
        error = evaluate_model(eig, test_parameters)
        accuracy = compute_accuracy(tol, error)
        results.accuracy.append(round(accuracy.item(), 3))
        results.failed.append(bool(accuracy - error_bound < 0))

    # Chech what,if any, models failed the requirements
    assert (
        sum(results.failed) == 0
    ), "Error bound not reached: Failed for matrix sizes {} with accuracy {}, resepectively".format(
        list(compress(supported_sizes, results.failed)),
        list(compress(results.accuracy, results.failed)),
    )


if __name__ == "__main__":
    test_eig()

