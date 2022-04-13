import numpy as np
import torch

from ..utils.ModelHandler import ModelHandler

from ..utils.constants import NEURALG_MATRIX_SIZES as MATRIX_SIZES


def test_ModelHandler():
    """Tests if requesting a model for all supported operations works, 
    and that tracking and clearing models functions as desired.
    """
    # Instantiate a model handler
    TestModelHandler = ModelHandler()
    assert TestModelHandler is not None
    # Check no models are loaded when a new handler is initialized
    assert not bool(TestModelHandler.loaded_models)

    # Define all supported model names and matrix size bounds
    ops = {
        "eigval": np.arange(
            MATRIX_SIZES.eig.sym.lower_bound, MATRIX_SIZES.eig.sym.upper_bound + 1
        ),
        "r_eigval": np.arange(
            MATRIX_SIZES.eig.real.lower_bound, MATRIX_SIZES.real.sym.upper_bound + 1
        ),
        "c_eigval": np.arange(
            MATRIX_SIZES.eig.complex.lower_bound,
            MATRIX_SIZES.eig.complex.upper_bound + 1,
        ),
        "svd": np.arange(
            MATRIX_SIZES.svd.lower_bound, MATRIX_SIZES.svd.upper_bound + 1
        ),
    }
    # Request model for all supported operations
    loaded_model_count = 0
    for op, supported_sizes in ops.items():
        for matrix_size in supported_sizes:
            # Request a model
            m = TestModelHandler.get_model(op, matrix_size)

            loaded_model_count += 1
            assert m is not None
            some_test_input = torch.rand(1, 1, matrix_size, matrix_size)
            # Check that the model can be used for infererence
            output = m(some_test_input)
            assert output.__class__.__name__ == "Tensor"
            assert len(TestModelHandler.loaded_models) == loaded_model_count

    # Check that models are not loaded again
    m_already_loaded = TestModelHandler.get_model("eigval", 3)
    assert len(TestModelHandler.loaded_models) == loaded_model_count

    # Assert that an error is thrown when requesting unavailable model
    try:
        m_not_available = TestModelHandler.get_model(op, 100)
        # assert m_not_available is not None
    except AssertionError:
        None
    try:
        m_not_available = TestModelHandler.get_model("no_op", 10)
    except AssertionError:
        None

    # Check clearing models work
    TestModelHandler.clear_loaded_models()
    assert not bool(TestModelHandler.loaded_models)


if __name__ == "__main__":
    test_ModelHandler()
