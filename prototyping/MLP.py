import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self,matrix_dimension,input_features,hidden_layers, n_neurons, activation = nn.ReLU(), output_activation = nn.Tanh()):
        super(MLP, self).__init__()
        self.matrix_dimension = matrix_dimension
        self.input_features = input_features
        self.n_neurons = n_neurons
        self.flatten = nn.Flatten(start_dim=2)
        self.unflatten = nn.Unflatten(-1,(matrix_dimension,matrix_dimension))
        self.net = []
        
        #Input layer
        self.net.append(nn.Linear(self.input_features,self.n_neurons))
        self.net.append(nn.ReLU())
        #Hidden layers
        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(self.n_neurons,self.n_neurons))
            self.net.append(activation)
        #Output layer 
        self.net.append(nn.Linear(self.n_neurons,self.matrix_dimension**2))
        #self.net.append(output_activation)
        
        self.net = nn.Sequential(*self.net)
    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        out = self.unflatten(x)
        return out

class EigMLP(nn.Module):
    def __init__(self,matrix_dimension,input_features,hidden_layers, n_neurons, activation = nn.ReLU()):
        super(EigMLP, self).__init__()
        self.matrix_dimension = matrix_dimension
        self.input_features = input_features
        self.n_neurons = n_neurons
        self.flatten = nn.Flatten(start_dim=2)
        self.unflatten = nn.Unflatten(-1,(matrix_dimension,matrix_dimension))
        self.net = []
        
        #Input layer
        self.net.append(nn.Linear(self.input_features,self.n_neurons))
        self.net.append(activation)
        #Hidden layers
        for i in range(hidden_layers-1):
            self.net.append(nn.Linear(self.n_neurons,self.n_neurons))
            self.net.append(activation)
        #Output layer 
        self.net.append(nn.Linear(self.n_neurons,self.matrix_dimension))
        
        self.net = nn.Sequential(*self.net)
    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        out = x 
        return out