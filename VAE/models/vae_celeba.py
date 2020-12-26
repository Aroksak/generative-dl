import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def _conv_layer_factory(self, input_channels, output_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, **kwargs),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.25)
        )

    def __init__(self, input_channels, bottleneck_dim):
        super().__init__()

        self.conv_0 = self._conv_layer_factory(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv_1 = self._conv_layer_factory(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv_2 = self._conv_layer_factory(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv_3 = self._conv_layer_factory(64, 64, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()

        self.mu = nn.Linear(8*8*64, bottleneck_dim)
        self.log_var = nn.Linear(8*8*64, bottleneck_dim)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def _conv_trans_layer_factory(self, input_channels, output_channels, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, **kwargs),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25)
        )

    def __init__(self, bottleneck_dim, output_channels):
        super().__init__()

        self.dense = nn.Linear(bottleneck_dim, 8*8*64)

        self.convtran_0 = self._conv_trans_layer_factory(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtran_1 = self._conv_trans_layer_factory(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtran_2 = self._conv_trans_layer_factory(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtran_3 = nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dense(x)
        x = F.leaky_relu(x)
        x = x.view(-1, 64, 8, 8)
        x = self.convtran_0(x)
        x = self.convtran_1(x)
        x = self.convtran_2(x)
        x = self.convtran_3(x)
        x = self.sigmoid(x)
        return x


class VariationalAutoEncoderCelebA(nn.Module):
    def __init__(self, input_channels=1, bottleneck_dim=2, output_channels=1):
        super().__init__()

        self.encoder = Encoder(input_channels=input_channels, bottleneck_dim=bottleneck_dim)
        self.decoder = Decoder(bottleneck_dim=bottleneck_dim, output_channels=output_channels)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        x = self.reparametrize(mu, log_var)
        x = self.decoder(x)
        return mu, log_var, x
