import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_0 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()

        self.conv_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(7*7*64, 2)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dense(x)
        # x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense = nn.Linear(2, 7*7*64)

        self.convtran_0 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()

        self.convtran_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtran_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtran_3 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)

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
        x = self.relu(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
