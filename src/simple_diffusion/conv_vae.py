import math

import torch
import torch.nn as nn


class NormalDistribution:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def log_prob(self, x):
        return -0.5 * (
            torch.log(2 * math.pi * self.var) + (x - self.mean) ** 2 / self.var
        ).sum(dim=1)

    def sample(self):
        return torch.normal(self.mean, torch.sqrt(self.var))


class ResNet2d(nn.Module):
    """This is taken from the VDVAE model"""

    def __init__(self, n_hidden, n_channels=None):
        super().__init__()
        if n_channels is None:
            n_channels = n_hidden
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, n_hidden, kernel_size=1, padding=0),
            nn.GroupNorm(1, n_hidden),
            nn.GELU(),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, n_hidden),
            nn.GELU(),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, n_hidden),
            nn.GELU(),
            nn.Conv2d(n_channels, n_hidden, kernel_size=1, padding=0),
        )

    def forward(self, x):
        y = self.net(x)
        return y + x


class ConvEncoder(nn.Module):
    """This is based off the VDVAE model"""

    def __init__(self, n_channels, latent_dim, hidden_dim, depth=3, n_resnet_blocks=1):
        super().__init__()
        blocks = []
        for i in range(depth):
            for _ in range(n_resnet_blocks):
                blocks.append(
                    ResNet2d(
                        hidden_dim * 2**i, n_channels=None if blocks else n_channels
                    )
                )
            blocks.append(nn.AvgPool2d(2))
        self.net = nn.Sequential(
            *blocks, nn.Flatten(), nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        y = self.net(x)
        return y


class ConvDecoder(nn.Module):
    """This is based off the VDVAE model"""

    def __init__(self, n_channels, latent_dim, hidden_dim, depth=3, n_resnet_blocks=1):
        super().__init__()
        self.in_linear = nn.Linear(latent_dim, hidden_dim)
        blocks = []
        for i in reversed(range(depth)):
            blocks.append(nn.UpsamplingNearest2d(2))
            for _ in range(n_resnet_blocks):
                blocks.append(
                    ResNet2d(
                        hidden_dim * 2**i,
                    )
                )
        self.net = nn.Sequential(
            *blocks, nn.Conv1d(hidden_dim, n_channels, kernel_size=2, padding=1)
        )

    def forward(self, x):
        x =
        y = self.net(x)
        return y


class ConvDecoder(nn.Module):
    """A simple fully connected denoiser network."""

    def __init__(self, n_channels, width, height, latent_dim, n_hidden=None):
        super().__init__()
        if n_hidden is None:
            n_hidden = latent_dim
        self.n_channels = n_channels
        self.width = width
        self.height = height
        self.mean_net = nn.Sequential(
            nn.Linear(latent_dim, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_channels * width * height),
        )

    def forward(self, x):
        x_ = x.view(x.shape[0], -1)
        mean = self.mean_net(x_).view(
            x.shape[0], self.n_channels, self.width, self.height
        )
        log_var = torch.zeros_like(mean)  # self.log_var_net(x_)
        return NormalDistribution(mean, 1e-2 * torch.exp(log_var))
