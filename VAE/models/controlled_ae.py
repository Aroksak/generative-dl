import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_0 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()

        self.conv_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        self.seed = nn.Linear(7*7*64, 1)
        self.ctrl = nn.Linear(7*7*64, 10)

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
        seed = self.seed(x)
        ctrls = F.softmax(self.ctrl(x), dim=-1)
        return seed, ctrls


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense = nn.Linear(11, 7*7*64)

        self.convtran_0 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()

        self.convtran_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtran_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtran_3 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)

    def forward(self, seed, ctrls):
        x = self.dense(torch.cat((seed, ctrls), dim=-1))
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


class ControlledAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        seed, ctrls = self.encoder(x)
        image = self.decoder(seed, ctrls)
        return ctrls, image
