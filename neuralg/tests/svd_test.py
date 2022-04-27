import numpy as np
from ..ops.svd import svd
from .test_utils import _test_op

from ..utils.constants import NEURALG_MATRIX_SIZES as MATRIX_SIZES


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
        MATRIX_SIZES.svd.lower_bound, MATRIX_SIZES.svd.upper_bound + 1
    )
    op = svd

    _test_op(op, supported_sizes, test_parameters, tol, error_bound)
