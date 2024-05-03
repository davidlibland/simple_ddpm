import math

import einops
import torch
from torch import nn as nn


class NormalDistribution:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def log_prob(self, x):
        return -0.5 * (
            torch.log(2 * math.pi * self.var) + (x - self.mean) ** 2 / self.var
        ).reshape([x.size(0), -1]).sum(dim=1)

    def sample(self):
        return torch.normal(self.mean, torch.sqrt(self.var))


class ResNetBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        dropout=0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.activation1 = nn.GELU()
        self.conv1 = nn.Linear(latent_dim, 2 * latent_dim)

        self.norm2 = nn.LayerNorm(2 * latent_dim)
        self.activation2 = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Linear(2 * latent_dim, latent_dim)
        # Initialize the final weights to zero:
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def __repr__(self):
        return f"ResNetBlock(n_channels={self.conv1.in_channels})"

    def forward(self, x):
        # Expand the input:
        h = self.norm1(x)
        h = self.activation1(h)
        h = self.conv1(h)

        # Collapse the input
        h = self.norm2(h)
        h = self.activation2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip Connection:
        h = h + x
        return h


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        n_layers=3,
        latent_dim=128,
        width=28,
        height=28,
        n_channels=1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.z_dim = latent_dim
        self.width = width
        self.height = height
        self.n_channels = n_channels

        self.encoder = nn.Sequential(
            nn.Linear(width * height * n_channels, hidden_size),
            nn.ReLU(),
            *[ResNetBlock(hidden_size) for _ in range(self.n_layers)],
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, x):
        x = einops.rearrange(x, "... d x y -> ... (d x y)")
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        n_layers=3,
        latent_dim=128,
        width=28,
        height=28,
        n_channels=1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.z_dim = latent_dim
        self.width = width
        self.height = height
        self.n_channels = n_channels

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            *[ResNetBlock(hidden_size) for _ in range(self.n_layers)],
            nn.Linear(hidden_size, width * height * n_channels),
        )

    def forward(self, z, cond=None):
        means = self.decoder(z)
        means = einops.rearrange(
            means,
            "... (d x y) -> ... d x y",
            x=self.height,
            y=self.width,
            d=self.n_channels,
        )
        log_var = torch.zeros_like(means)
        return NormalDistribution(means, 1e-2 * torch.exp(log_var))
