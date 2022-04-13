import numpy as np
from ..ops.eig import eig
from .test_utils import _test_op

from ..utils.constants import NEURALG_MATRIX_SIZES as MATRIX_SIZES


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
            MATRIX_SIZES.eig.sym.lower_bound, MATRIX_SIZES.eig.sym.upper_bound + 1,
        )
        test_parameters[
            "wigner"
        ] = True  # Test performance on Wigner matrices with eigvals with variance as in training.
    elif real:
        # Mock limits, very very high
        error_bound = 0.7
        tol = 0.38  # Preliminary tolerance
        supported_sizes = np.arange(
            MATRIX_SIZES.eig.real.lower_bound, MATRIX_SIZES.eig.real.upper_bound + 1
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
            MATRIX_SIZES.svd.lower_bound, MATRIX_SIZES.svd.upper_bound + 1
        )  # Test performance on random matrices with uniformly distributed elements

    op = lambda x: eig(x, symmetric=symmetric, real=real)

    _test_op(op, supported_sizes, test_parameters, tol, error_bound)


if __name__ == "__main__":
    test_eig()
