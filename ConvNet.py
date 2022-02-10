import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvNet(nn.Module):
    def __init__(self, hidden_layers, filters, kernel_size):
        super(ConvNet, self).__init__()
        self.net = []
        self.net.append(nn.Conv2d(1,filters,kernel_size, padding = "same"))
        for i in range(hidden_layers):
            self.net.append(nn.BatchNorm2d(filters))
            self.net.append(nn.Conv2d(filters,filters,kernel_size, padding = "same"))
            self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(filters,1,kernel_size, padding = "same"))
        
        self.net = nn.Sequential(*self.net)
    def forward(self, x):
        out = self.net(x)
        return out