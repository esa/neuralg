# From https://github.com/darioizzo/geodesyNets/blob/master/gravann/networks/_nerf.py,
# Implementation of architecture from "NeRF: Representing Scenes as
# Neural Radiance Fields for View Synthesis" , https://arxiv.org/pdf/2003.08934.pdf
import torch
import torch.nn as nn


class NERFLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=nn.ReLU()):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.activation = activation

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.uniform_(self.linear.bias.data, -0.0, 0.0)

    def forward(self, input):
        return self.activation(self.linear(input))


class EigNERF(nn.Module):
    def __init__(
        self,
        matrix_dimension,
        in_features,
        skip=[2, 4, 6],
        n_neurons=200,
        out_features=None,
        hidden_layers=8,
    ):
        super().__init__()
        self.model_type = "nerf"
        self.matrix_dimension = matrix_dimension
        self.in_features = in_features
        self.skip = skip

        if out_features is not None:
            self.out_features = out_features
        else:
            self.out_features = matrix_dimension  # For e.g. full eigval problem

        self.net = nn.ModuleList()
        # Added this for more robust against different batch dimensions
        self.flatten_batch = nn.Flatten(start_dim=0, end_dim=-3)
        self.flatten = nn.Flatten(start_dim=-2)

        self.net.append(NERFLayer(in_features, n_neurons))

        for i in range(hidden_layers):
            if i in self.skip:
                self.net.append(NERFLayer(n_neurons + in_features, n_neurons))
            else:
                self.net.append(NERFLayer(n_neurons, n_neurons))

        self.net.append(nn.Linear(n_neurons, matrix_dimension))

    def forward(self, x):
        batch_dim = x.shape[0:-2]

        # Is there a smarter way to deal with different batch dimensions?
        if len(batch_dim) > 1:
            x = self.flatten_batch(x)

        x_flat = self.flatten(x)

        # save for skip connection
        identity = x_flat

        # compute first layer
        out = self.net[0].forward(x_flat)

        # compute all other layers and apply skip where requested
        for layer_idx in range(1, len(self.net)):
            out = self.net[layer_idx].forward(out)
            if layer_idx in self.skip:
                out = torch.cat([out, identity], dim=-1)
        if len(batch_dim) > 1:
            out = nn.Unflatten(0, batch_dim)(out)
        return out


class CEigNERF(nn.Module):
    def __init__(
        self,
        matrix_dimension,
        in_features,
        skip=[2, 4, 6],
        n_neurons=200,
        out_features=None,
        hidden_layers=8,
    ):
        super().__init__()
        self.model_type = "complex_nerf"
        self.matrix_dimension = matrix_dimension
        self.in_features = in_features
        self.skip = skip

        if out_features is not None:
            self.out_features = out_features
        else:
            self.out_features = matrix_dimension  # For e.g. full eigval problem

        self.net = nn.ModuleList()

        self.flatten_batch = nn.Flatten(start_dim=0, end_dim=-3)
        self.flatten = nn.Flatten(start_dim=-2)

        self.net.append(NERFLayer(in_features, n_neurons))

        for i in range(hidden_layers):
            if i in self.skip:
                self.net.append(NERFLayer(n_neurons + in_features, n_neurons))
            else:
                self.net.append(NERFLayer(n_neurons, n_neurons))

        self.net.append(nn.Linear(n_neurons, 2 * matrix_dimension))

    def forward(self, x):
        batch_dim = x.shape[0:-2]

        # Is there a smarter way to deal with different batch dimensions?
        if len(batch_dim) > 1:
            x = self.flatten_batch(x)

        x_flat = self.flatten(x)

        # save for skip connection
        identity = x_flat

        # compute first layer
        out = self.net[0].forward(x_flat)

        # compute all other layers and apply skip where requested
        for layer_idx in range(1, len(self.net)):
            out = self.net[layer_idx].forward(out)
            if layer_idx in self.skip:
                out = torch.cat([out, identity], dim=-1)
        if len(batch_dim) >= 1:
            out = nn.Unflatten(0, batch_dim)(out)
            out = nn.Unflatten(-1, ([2, -1]))(out)
        else:
            out = nn.Unflatten(0, ([2, -1]))(out)
        # Should one do this here, or keep everything real when computing the loss?
        re_out, im_out = out[:, :, 0], out[:, :, 1]
        return torch.complex(re_out, im_out)

