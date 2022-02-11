import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self,matrix_dimension,hidden_layers, n_neurons activation = nn.ReLU(), output_activation = nn.Tanh()):
        super(MLP, self).__init__()
        self.matrix_dimension = matrix_dimension
        self.flatten = nn.Flatten(start_dim=2)
        self.unflatten = nn.Unflatten(-1,(matrix_dimension,matrix_dimension))
        self.net = []
        
        #Input layer
        self.net.append(nn.Linear(self.matrix_dimension**2,n_neurons))
        self.net.append(nn.ReLU())
        #Hidden layers
        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(self.matrix_dimension**2,n_neurons))
            self.net.append(activation)
        #Output layer 
        self.net.append(nn.Linear(self.matrix_dimension**2,self.matrix_dimension**2))
        #self.net.append(output_activation)
        
        self.net = nn.Sequential(*self.net)
    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        out = self.unflatten(x)
        return out