from cgi import test
from json import load
import numpy as np
import torch

from neuralg.io.load_model import load_model
from neuralg.models.TestModel import TestModel
from neuralg.utils.ModelHandler import ModelHandler


def test_ModelHandler():
    # Instantiate a model handler
    TestModelHandler = ModelHandler()
    assert TestModelHandler is not None
    # Check no models are loaded when a new handler is initialized
    assert not bool(TestModelHandler.loaded_models)
    # Request model for all supported operations
    # Right now that is computing eigenvalues for matrices from 3x3 up to 10x10
    supported_sizes = np.arange(3, 11)
    ops = ["eigval"]
    loaded_model_count = 0
    for op in ops:
        for matrix_size in supported_sizes:
            # Request a model
            m = TestModelHandler.ship_model(op, matrix_size)

            loaded_model_count += 1
            assert m is not None
            some_test_input = torch.rand(1, 1, matrix_size, matrix_size)
            # Check that the model can be used for infererence
            output = m(some_test_input)
            assert output.__class__.__name__ == "Tensor"
            assert len(TestModelHandler.loaded_models) == loaded_model_count

    # (Try) to check that models are not loaded again
    m_already_loaded = TestModelHandler.ship_model("eigval", 3)
    assert len(TestModelHandler.loaded_models) == loaded_model_count

    # Assert that a value error is thrown when requesting unavailable model
    try:

        m_not_available = TestModelHandler.ship_model(op, 100)
        # assert m_not_available is not None
    except ValueError:
        None
    try:
        m_not_available = TestModelHandler.ship_model("no_op", 10)
    except ValueError:
        None


if __name__ == "__main__":
    test_ModelHandler()
