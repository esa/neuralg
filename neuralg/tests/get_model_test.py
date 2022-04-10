from hashlib import new
import torch
from ..io.get_model import get_model
from ..utils.constants import NEURALG_SUPPORTED_OPERATIONS


def test_get_model():
    matrix_dimension = 5  # Test that it works for an all-supported size
    test_input = torch.rand(10, 1, matrix_dimension, matrix_dimension)
    for operation in NEURALG_SUPPORTED_OPERATIONS:
        models = []
        get_model_handle = lambda symmetric, real: get_model(
            operation, matrix_dimension, symmetric, real
        )
        models.append(get_model_handle(False, False))
        models.append(get_model_handle(True, False))
        models.append(get_model_handle(False, True))
        for model in models:
            assert model is not None
            out = model(test_input)
            assert out.requires_grad

        new_model = get_model(
            operation, 100, new=True
        )  # Test instantiating a new model
        assert new_model is not None

