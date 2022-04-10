import torch
from ..io.load_model import load_model
from ..utils.count_parameters import count_parameters


def test_load_model():
    """Tests if models can be loaded succesfully
    """
    # Symmetric matrices
    model_name = "eigval3"
    model = load_model(model_name)
    _test_model(model, "EigNERF", "nerf")
    assert count_parameters(model) == 329603

    # Non-symmetric matrices with real eigenvalues
    model_name = "r_eigval3"
    model = load_model(model_name)
    _test_model(model, "EigNERF", "nerf")

    # Non-symmetric matrices with potentially complex eigenvalues
    model_name = "c_eigval3"
    model = load_model(model_name)

    _test_model(model, "CEigNERF", "complex_nerf")

    # svd operation
    model_name = "svd3"
    model = load_model(model_name)
    _test_model(model, "EigNERF", "nerf")


def _test_model(model, name, type):
    """ Check that model has correct name, type and correct output format.

    Args:
        model (torch.nn): Model to be tested
        name (str): Name of loaded model
        type (str): Type of loaded model 
    """
    some_test_input = torch.rand(1, 1, 3, 3)
    assert model is not None
    assert model.__class__.__name__ == name
    assert model.model_type == type

    output = model(some_test_input)
    assert output.__class__.__name__ == "Tensor"
    assert output.shape == torch.Size([1, 1, 3])


if __name__ == "__main__":
    test_load_model()
