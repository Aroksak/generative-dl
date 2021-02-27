import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    @staticmethod
    def _conv_layer_factory(input_channels, output_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, **kwargs),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()

        self.conv1 = self._conv_layer_factory(input_channels, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = self._conv_layer_factory(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = self._conv_layer_factory(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv4 = self._conv_layer_factory(128, 128, kernel_size=5, stride=1, padding=2)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dense(x)
        return x


class Generator(nn.Module):
    @staticmethod
    def _conv_layer_factory(input_channels, output_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, **kwargs),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def __init__(self):
        super(Generator, self).__init__()

        self.dense = nn.Sequential(
            nn.Linear(100, 3136),
            nn.BatchNorm1d(3136),
            nn.ReLU()
        )

        self.conv1 = self._conv_layer_factory(64, 128, kernel_size=5, padding=2)
        self.conv2 = self._conv_layer_factory(128, 64, kernel_size=5, padding=2)
        self.conv3 = self._conv_layer_factory(64, 64, kernel_size=5, padding=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 64, 7, 7)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
