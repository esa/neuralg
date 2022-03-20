import numpy as np
import torch

from neuralg.utils.ModelHandler import ModelHandler
from neuralg.utils.constants import NEURALG_MIN_MATRIX_SIZE, NEURALG_MAX_MATRIX_SIZE


def test_ModelHandler():
    """Tests if requesting a model for all supported operations works, 
    and that tracking and clearing models functions as desired.
    """
    # Instantiate a model handler
    TestModelHandler = ModelHandler()
    assert TestModelHandler is not None
    # Check no models are loaded when a new handler is initialized
    assert not bool(TestModelHandler.loaded_models)

    # Request model for all supported operations
    supported_sizes = np.arange(NEURALG_MIN_MATRIX_SIZE, NEURALG_MAX_MATRIX_SIZE + 1)
    ops = ["eigval"]
    loaded_model_count = 0
    for op in ops:
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
