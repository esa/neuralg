import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.norm = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(1, 64, 3, padding="same")
        self.conv2 = nn.Conv2d(64, 64, 3, padding="same")
        self.conv3 = nn.Conv2d(64, 64, 3, padding="same")
        self.conv4 = nn.Conv2d(64, 64, 3, padding="same")
        self.conv5 = nn.Conv2d(64, 64, 3, padding="same")
        self.conv6 = nn.Conv2d(64, 1, 3, padding="same")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm(x)
        x = F.relu(self.conv2(x))
        x = self.norm(x)
        x = F.relu(self.conv3(x))
        x = self.norm(x)
        x = F.relu(self.conv4(x))
        x = self.norm(x)
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x