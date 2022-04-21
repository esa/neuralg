import torch
from dotmap import DotMap
from itertools import compress
from ..evaluation.evaluate_model import evaluate_model
from ..evaluation.compute_accuracy import compute_accuracy


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
    """Check operations support for different types of input e.g. in and out of batch mode with different batch dimensions

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
