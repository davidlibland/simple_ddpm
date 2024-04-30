import math

import einops
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


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

    def __init__(self, n_hidden, total_depth):
        super().__init__()
        self.res_net = nn.Sequential(
            nn.Conv2d(n_hidden, n_hidden, kernel_size=1, padding=0),
            nn.GroupNorm(1, n_hidden),
            nn.GELU(),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, n_hidden),
            nn.GELU(),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, n_hidden),
            nn.GELU(),
            nn.Conv2d(n_hidden, n_hidden, kernel_size=1, padding=0),
        )
        # Scale the output of the blocks by sqrt(1 / total_depth):
        self.res_net[-1].weight.data *= math.sqrt(1 / total_depth)
        # Initialize the last layer to zero:
        self.res_net[-1].bias.data.zero_()

    def forward(self, x):
        y = self.res_net(x)
        return y + x


class ConvEncoder(nn.Module):
    """This is based off the VDVAE model"""

    def __init__(
        self,
        n_channels,
        height,
        width,
        latent_dim,
        hidden_dim,
        depth=3,
        n_resnet_blocks=1,
    ):
        super().__init__()
        blocks = [nn.Conv2d(n_channels, hidden_dim, kernel_size=3, padding=1)]
        for i in range(depth):
            for j in range(n_resnet_blocks):
                blocks.append(
                    ResNet2d(
                        hidden_dim,
                        total_depth=depth * n_resnet_blocks,
                    )
                )
            blocks.append(nn.AvgPool2d(2))
            height //= 2
            width //= 2
        self.height = height
        self.width = width
        self.enc_net = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden_dim * height * width, latent_dim)

    def forward(self, x):
        y = self.enc_net(x)
        y = einops.rearrange(y, "... c h w -> ... (c h w)")
        return self.out(y)


class ConvDecoder(nn.Module):
    """This is based off the VDVAE model"""

    def __init__(
        self,
        n_channels,
        height,
        width,
        latent_dim,
        hidden_dim,
        depth=3,
        n_resnet_blocks=1,
    ):
        super().__init__()
        self.top_height = height // 2**depth
        self.top_width = width // 2**depth
        self.hidden_dim = hidden_dim
        self.in_linear = nn.Linear(
            latent_dim, hidden_dim * self.top_height * self.top_width
        )
        blocks = []
        for _ in reversed(range(depth)):
            blocks.append(nn.UpsamplingNearest2d(scale_factor=2))
            for _ in range(n_resnet_blocks):
                blocks.append(
                    ResNet2d(
                        hidden_dim,
                        total_depth=depth * n_resnet_blocks,
                    )
                )
        self.dec_net = nn.Sequential(
            *blocks,
            nn.Conv2d(hidden_dim, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        y = self.in_linear(x)
        y = einops.rearrange(
            y,
            "... (c h w) -> ... c h w",
            w=self.top_width,
            h=self.top_height,
            c=self.hidden_dim,
        )

        y = self.dec_net(y)
        log_var = torch.zeros_like(y)
        return NormalDistribution(y, 1e-2 * torch.exp(log_var))
