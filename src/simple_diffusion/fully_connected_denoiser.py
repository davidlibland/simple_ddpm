import torch
from torch import nn as nn


class Denoiser(nn.Module):
    """A simple fully connected denoiser network."""

    def __init__(self, latent_shape, time_scale, n_freqs=32, n_hidden=64):
        super().__init__()
        n_dims = torch.prod(torch.tensor(latent_shape)).item()
        self.net = nn.Sequential(
            nn.Linear(n_dims + n_freqs, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_dims),
        )
        self.register_buffer(
            "fourier_freqs",
            torch.arange(1, n_freqs, 2).float() / (2 * n_freqs) / time_scale,
        )

    def forward(self, x, t):
        x_ = x.view(x.shape[0], -1)
        y = self.net(torch.cat([x_, self.encode_time(t)], dim=-1))
        return y.view(x.shape)

    def encode_time(self, t):
        return torch.cat(
            [
                torch.sin(t * self.fourier_freqs * 2 * torch.pi),
                torch.cos(t * self.fourier_freqs * 2 * torch.pi),
            ],
            dim=-1,
        )
