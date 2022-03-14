import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNET, self).__init__()
        self.conv1 = self.contract_block(in_channels, 32, 3, 1)
        # self.conv2 = self.contract_block(32, 64, 3, 1)
        # self.conv3 = self.contract_block(64, 128, 3, 1)
        # self.upconv3 = self.expand_block(128, 64, 3, 1)
        # self.upconv2 = self.expand_block(64, 32, 3, 1)
        self.upconv1 = self.expand_block(32, out_channels, 3, 1)

    def forward(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        # conv2 = self.conv2(conv1)
        # conv3 = self.conv3(conv2)

        # upconv3 = self.upconv3(conv3)
        # upconv2 = self.upconv2(conv2)
        # upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        # upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        upconv1 = self.upconv1(conv1)
        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )
        return expand
