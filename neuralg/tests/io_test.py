import torch
from neuralg.io.load_model import load_model
from neuralg.utils.count_parameters import count_parameters


def test_load_model():
    """Tests if a model can be loaded succesfully
    """
    model_name = "eigval3"
    model = load_model(model_name)

    assert model is not None
    assert model.__class__.__name__ == "EigNERF"
    assert model.model_type == "nerf"

    assert count_parameters(model) == 329603

    some_test_input = torch.rand(1, 1, 3, 3)

    output = model(some_test_input)
    assert output.__class__.__name__ == "Tensor"
    assert output.shape == torch.Size([1, 1, 3])


if __name__ == "__main__":
    test_load_model()
