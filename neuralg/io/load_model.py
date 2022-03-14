import torch
import pickle as pk
import numpy as np
import sys
from neuralg.models.TestModel import TestModel
from neuralg.models.nerf import EigNERF


""" TODO
[] Fix the paths to saved models
[] Eventually figure out how to keep track of available models in a smarter way
"""

# For now, avaiable models to load are just stored locally in this dict
# Maybe it makes more sense to have this in models folder?
available_models_dict = {
    "TestModel": ["./neuralg/tests/test_data/test_model.pt", TestModel()]
}
# Added some trained nerf models for first minimal module
for d in range(3, 11):
    available_models_dict["eigval{}".format(d)] = [
        "./neuralg/models/saved_models/eigval{}.pt".format(d),
        EigNERF(d, d ** 2),
    ]


def load_model(model_name):

    assert model_name in available_models_dict

    [model_path, model] = available_models_dict[model_name]
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    return model
