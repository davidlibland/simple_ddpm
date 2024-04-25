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


class Encoder(nn.Module):
    """A simple fully connected denoiser network."""

    def __init__(self, data_dim, latent_dim, n_hidden=None):
        super().__init__()
        if n_hidden is None:
            n_hidden = latent_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, latent_dim),
        )

    def forward(self, x):
        x_ = x.view(x.shape[0], -1)
        y = self.net(x_)
        return y


class Decoder(nn.Module):
    """A simple fully connected denoiser network."""

    def __init__(self, latent_dim, data_dim, n_hidden=None):
        super().__init__()
        if n_hidden is None:
            n_hidden = latent_dim
        self.mean_net = nn.Sequential(
            nn.Linear(latent_dim, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, data_dim),
        )
        # self.log_var_net = nn.Sequential(
        #     nn.Linear(latent_dim, n_hidden),
        #     nn.LayerNorm(n_hidden),
        #     nn.ReLU(),
        #     nn.Linear(n_hidden, data_dim),
        # )

    def forward(self, x):
        x_ = x.view(x.shape[0], -1)
        mean = self.mean_net(x_)
        log_var = torch.zeros_like(mean)  # self.log_var_net(x_)
        return NormalDistribution(mean, 1e-2 * torch.exp(log_var))
