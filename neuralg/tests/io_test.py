import torch
import sys

from neuralg.io.load_model import load_model


# from utils import countParameters
def test_load_model():

    model_name = "TestModel"
    model = load_model(model_name)

    assert model is not None
    assert model.__class__.__name__ == "TestModel"
    assert model.model_type == "ConvNet"

    assert count_parameters(model) == 2689

    some_test_input = torch.rand(1, 1, 10, 10)

    output = model(some_test_input)
    assert output.__class__.__name__ == "Tensor"
    assert output.shape == torch.Size([1, 1, 10])


def count_parameters(model):
    """Counts number of trainable parameters in torch model

    Args:
        model (_type_): Trained model

    Returns:
        int : Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    test_load_model()
