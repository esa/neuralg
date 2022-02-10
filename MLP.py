#Simple MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.unflatten = nn.Unflatten(-1,(3,3))
        self.relu_stack = nn.Sequential(
                nn.Linear(9,9),
                nn.ReLU(),
                nn.Linear(9,9),
                nn.ReLU(),
                nn.Linear(9,9),
                nn.ReLU(),
                nn.Linear(9,9),
                nn.ReLU(),
                nn.Linear(9,9),
                nn.ReLU(),
                nn.Linear(9,9),
                nn.ReLU(),
                nn.Linear(9,9),
                nn.ReLU(),
                nn.Linear(9,9),)
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu_stack(x)
        x = self.unflatten(x)
        return x