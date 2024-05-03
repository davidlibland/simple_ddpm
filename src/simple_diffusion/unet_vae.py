import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

N_GROUPS = 1


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


class StackPool(nn.Module):
    """Downsamples the input by stacking the channels."""

    def __init__(self, h=2, w=2):
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x):
        return einops.rearrange(
            x, "b c (h h1) (w w1) -> b (h1 w1 c) h w", h1=self.h, w1=self.w
        )


class UnStackUpsample(nn.Module):
    """Upsamples the input by unstacking the channels."""

    def __init__(self, h=2, w=2):
        super().__init__()
        self.h = h
        self.w = w

    def forward(self, x):
        return einops.rearrange(x, "b (h1 w1 c) h w -> b c (h h1) (w w1)", h1=2, w1=2)


class ChannelAttention(nn.Module):
    def __init__(
        self, in_channels, out_channels, n_heads=8, n_freqs=32, n_groups=N_GROUPS
    ):
        super().__init__()
        if in_channels % n_heads != 0:
            raise ValueError(
                f"Number of heads {n_heads} must divide number of channels {out_channels}"
            )
        self.norm = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)
        self.conv = nn.Conv2d(
            in_channels + 2 * n_freqs, out_channels * 3, kernel_size=1
        )
        self.scale = in_channels**-0.5
        self.n_heads = n_heads
        self.register_buffer(
            "fourier_freqs",
            torch.arange(1, n_freqs, 2).float() / (2 * n_freqs),
        )

    def forward(self, x):
        x_norm = self.norm(x)
        w = x.size(-1)
        h = x.size(-2)
        tw = torch.arange(w, device=x.device).float() / w
        th = torch.arange(h, device=x.device).float() / h
        th = self.encode_time(th.view([1, 1, -1, 1])).expand(
            x.size(0), -1, -1, x.size(-1)
        )
        tw = self.encode_time(tw.view([1, 1, 1, -1])).expand(
            x.size(0), -1, x.size(-2), -1
        )
        x_loc_emb = torch.cat(
            [x_norm, th, tw],
            dim=1,
        )
        qkv = self.conv(x_loc_emb)
        qkv = torch.chunk(qkv, 3, dim=1)
        q, k, v = [
            einops.rearrange(
                t,
                "b (heads c) h w -> b heads (h w) c",
                heads=self.n_heads,
            )
            for t in qkv
        ]
        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = einops.rearrange(
            attn, "b heads (h w) c -> b (heads c) h w", h=x.size(-2), w=x.size(-1)
        )
        return attn

    def encode_time(self, t):
        return torch.cat(
            [
                torch.sin(t * self.fourier_freqs.view([1, -1, 1, 1]) * 2 * torch.pi),
                torch.cos(t * self.fourier_freqs.view([1, -1, 1, 1]) * 2 * torch.pi),
            ],
            dim=1,
        )


class UnetEncoder(nn.Module):
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
        if latent_dim % (depth + 1) != 0:
            raise ValueError(
                f"Latent dimension {latent_dim} must be divisible by depth + 1 {depth + 1}"
            )
        blocks = [nn.Conv2d(n_channels, hidden_dim, kernel_size=3, padding=1)]
        connections = []
        for i in range(depth):
            stack = []
            for _ in range(n_resnet_blocks):
                stack.append(
                    ResNet2d(
                        hidden_dim,
                        total_depth=depth * n_resnet_blocks,
                    )
                )
            stack = nn.Sequential(*stack)
            blocks.append(stack)
            height //= 2
            width //= 2
            connections.append(
                # nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    latent_dim // (depth + 1),
                    kernel_size=2,
                    padding=0,
                    stride=2,
                )
                # nn.Flatten(1),
                # nn.LayerNorm(hidden_dim * height * width),
                # nn.GELU(),
                # nn.Linear(hidden_dim * height * width, latent_dim // (depth + 1)),
                # )
            )
        self.blocks = nn.ModuleList(blocks)
        self.connections = nn.ModuleList(connections)
        self.height = height
        self.width = width
        self.out = nn.Linear(hidden_dim * height * width, latent_dim // (depth + 1))

    def forward(self, x):
        z = []
        for block, connector in zip(self.blocks, self.connections):
            x = block(x)
            z.append(connector(x).mean(dim=(2, 3)))
            x = F.avg_pool2d(x, 2)
        z.append(self.out(einops.rearrange(x, "... c h w -> ... (c h w)")))
        return torch.cat(z, dim=1)


class UnetDecoder(nn.Module):
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
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.in_linear = nn.Linear(
            latent_dim // (depth + 1), hidden_dim * self.top_height * self.top_width
        )
        blocks = []
        connectors = []
        for i in reversed(range(depth)):
            connectors.append(
                nn.Sequential(
                    nn.LayerNorm(latent_dim // (depth + 1)),
                    nn.GELU(),
                    nn.Linear(
                        latent_dim // (depth + 1),
                        hidden_dim,
                    ),
                )
            )
            # initialize the connector to zero:
            connectors[-1][-1].weight.data.zero_()
            connectors[-1][-1].bias.data.zero_()
            stack = []
            for _ in range(n_resnet_blocks):
                stack.append(
                    ResNet2d(
                        hidden_dim,
                        total_depth=depth * n_resnet_blocks,
                    )
                )
            stack = nn.Sequential(*stack)
            blocks.append(stack)
        self.blocks = nn.ModuleList(blocks)
        self.connectors = nn.ModuleList(connectors)
        self.out = nn.Conv2d(hidden_dim, n_channels, kernel_size=3, padding=1)

    def forward(self, x):
        zs = torch.chunk(x, self.depth + 1, dim=1)
        y = self.in_linear(zs[0])
        width = self.top_width
        height = self.top_height
        y = einops.rearrange(
            y,
            "... (c h w) -> ... c h w",
            w=width,
            h=height,
            c=self.hidden_dim,
        )
        for z_, block, connector in zip(zs[1:], self.blocks, self.connectors):
            y = F.upsample_bilinear(y, scale_factor=2)
            width *= 2
            height *= 2
            z_ = einops.rearrange(
                connector(z_),
                "... c -> ... c 1 1",
                # w=width,
                # h=height,
                c=self.hidden_dim,
            )
            y = y + z_
            y = block(y)

        mean = self.out(y)
        log_var = torch.zeros_like(mean)
        return NormalDistribution(mean, 1e-2 * torch.exp(log_var))
