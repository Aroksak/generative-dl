import torch
import torch.nn as nn


class Encoder(nn.Module):
    def _conv_layer_factory(self, input_channels, output_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, **kwargs),
            nn.LeakyReLU(),
        )

    def __init__(self, input_channels=1, bottleneck_dim=2):
        super().__init__()

        self.conv_0 = self._conv_layer_factory(input_channels, 32, kernel_size=3, padding=1)

        self.conv_1 = self._conv_layer_factory(32, 64, kernel_size=3, stride=2, padding=1)

        self.conv_2 = self._conv_layer_factory(64, 64, kernel_size=3, stride=2, padding=1)

        self.conv_3 = self._conv_layer_factory(64, 64, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        self.mu = nn.Linear(7*7*64, bottleneck_dim)
        self.log_var = nn.Linear(7*7*64, bottleneck_dim)

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
    def __init__(self, bottleneck_dim=2, output_channels=1):
        super().__init__()

        self.dense = nn.Linear(bottleneck_dim, 7*7*64)

        self.convtran_0 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()

        self.convtran_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtran_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtran_3 = nn.ConvTranspose2d(32, output_channels, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        x = x.view(-1, 64, 7, 7)
        x = self.convtran_0(x)
        x = self.relu(x)
        x = self.convtran_1(x)
        x = self.relu(x)
        x = self.convtran_2(x)
        x = self.relu(x)
        x = self.convtran_3(x)
        x = self.sigmoid(x)
        return x


class VariationalAutoEncoder(nn.Module):
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
