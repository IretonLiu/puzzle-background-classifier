import torch.nn as nn
import torch.nn.functional as F


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConvolution(in_channels, out_channels)


class UNet(nn.Module):
    """
    Pytorch implementation of UNet, following the same architecture as
    the on presented in the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation".
    """

    def __init__(self):
        super().__init__()

        # Contracting path

        self.conv1 = DoubleConvolution(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
