from torch import nn
import torch
import numpy as np

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
class Siren(nn.Module):
    def __init__(self, matrix_dimension, hidden_features, hidden_layers, outermost_linear=True, outermost_activation=nn.Tanh(),
                 first_omega_0=30, hidden_omega_0=30.):

        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.unflatten = nn.Unflatten(-1,(matrix_dimension,matrix_dimension))
        self.net = []
        self.net.append(SineLayer(matrix_dimension**2, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, matrix_dimension**2)

            self.net.append(final_linear)
            #self.net.append(outermost_activation)
        else:
            self.net.append(SineLayer(hidden_features, matrix_dimension**2,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        output = self.unflatten(x)
        return output

class EigSiren(nn.Module):
    def __init__(self, matrix_dimension, hidden_features, hidden_layers, outermost_linear=True, outermost_activation=nn.Tanh(),
                 first_omega_0=1, hidden_omega_0=1.):

        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.net = []
        self.net.append(SineLayer(matrix_dimension**2, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, matrix_dimension)

            self.net.append(final_linear)
            #self.net.append(outermost_activation)
        else:
            self.net.append(SineLayer(hidden_features, matrix_dimension**2,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        output = x
        return output