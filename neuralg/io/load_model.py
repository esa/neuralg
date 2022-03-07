import torch
import pickle as pk
import numpy as np
import sys
import neuralg.models.TestModel as TestModel


def load_model(model_name):  # or should it be model path?
    # need both path and model class to load a model
    available_models_dict = {
        "TestModel": ["neuralg/tests/test_data/test_model.pt", TestModel()]
    }
    assert model_name in available_models_dict
    [model_path, model] = available_models_dict["model_name"]
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
