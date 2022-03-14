import torch
from torch import nn


class TestModel(nn.Module):
    def __init__(self, hidden_layers=2, filters=16, kernel_size=3):
        super(TestModel, self).__init__()
        self.model_type = "ConvNet"
        self.net = []
        self.net.append(nn.Conv2d(1, filters, kernel_size, padding="same"))
        self.net.append(nn.BatchNorm2d(filters))
        self.net.append(nn.ReLU())
        for i in range(hidden_layers - 1):
            self.net.append(nn.Conv2d(filters, filters, kernel_size, padding="same"))
            self.net.append(nn.BatchNorm2d(filters))
            self.net.append(nn.ReLU())

        self.net.append(nn.Conv2d(filters, 1, kernel_size, padding="same"))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        out = self.net(x)
        out = torch.diagonal(out, 0, 2, 3)
        return out
