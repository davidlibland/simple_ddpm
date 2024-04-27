import math

import torch
import torch.nn as nn

import einops


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


class ConvEncoder(nn.Module):
    """A simple fully connected denoiser network."""

    def __init__(self, n_channels, width, height, latent_dim, n_hidden=None):
        super().__init__()
        if n_hidden is None:
            n_hidden = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, n_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, n_hidden),
            nn.ReLU(),
            nn.Conv2d(n_hidden, latent_dim, kernel_size=3, padding=1),
        )
        self.width = width
        self.height = height
        self.out = nn.Linear(latent_dim * width * height, latent_dim)

    def forward(self, x):
        y = self.net(x)
        y = einops.rearrange(y, "b c h w -> b (c h w)")
        y = self.out(y)
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
