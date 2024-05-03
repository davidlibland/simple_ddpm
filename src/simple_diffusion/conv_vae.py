import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

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


class FlattenAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        lookup_dim=None,
        n_heads=8,
        n_freqs=32,
        n_groups=N_GROUPS,
    ):
        super().__init__()
        if out_channels % n_heads != 0:
            raise ValueError(
                f"Number of heads {n_heads} must divide number of channels {out_channels}"
            )
        out_channels_per_head = out_channels // n_heads
        if lookup_dim is None:
            lookup_dim = in_channels
        self.norm = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)
        self.kconv = nn.Conv2d(
            in_channels + 2 * n_freqs, lookup_dim * n_heads, kernel_size=1
        )
        self.vconv = nn.Conv2d(
            in_channels + 2 * n_freqs, out_channels_per_head * n_heads, kernel_size=1
        )
        self.query = nn.Parameter(torch.randn(1, n_heads, 1, lookup_dim))
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
        k = einops.rearrange(
            self.kconv(x_loc_emb),
            "b (heads c) h w -> b heads (h w) c",
            heads=self.n_heads,
        )
        v = einops.rearrange(
            self.vconv(x_loc_emb),
            "b (heads c) h w -> b heads (h w) c",
            heads=self.n_heads,
        )
        attn = F.scaled_dot_product_attention(self.query, k, v)
        return attn.reshape(x.size(0), -1)

    def encode_time(self, t):
        return torch.cat(
            [
                torch.sin(t * self.fourier_freqs.view([1, -1, 1, 1]) * 2 * torch.pi),
                torch.cos(t * self.fourier_freqs.view([1, -1, 1, 1]) * 2 * torch.pi),
            ],
            dim=1,
        )


class UnFlattenAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        height,
        n_chunks=None,
        lookup_dim=None,
        n_heads=8,
        n_freqs=32,
        n_groups=N_GROUPS,
    ):
        super().__init__()
        if out_channels % n_heads != 0:
            raise ValueError(
                f"Number of heads {n_heads} must divide number of channels {out_channels}"
            )
        out_channels_per_head = out_channels // n_heads
        if lookup_dim is None:
            lookup_dim = in_channels
        if n_chunks is None:
            n_chunks = height * width
        self.norm = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)
        self.klin = nn.Linear(in_channels, lookup_dim * n_heads * n_chunks)
        self.vlin = nn.Linear(in_channels, out_channels_per_head * n_heads * n_chunks)
        self.qconv_pos = nn.Conv2d(2 * n_freqs, lookup_dim * n_heads, kernel_size=1)
        self.qlin_input = nn.Linear(in_channels, lookup_dim * n_heads)
        self.n_heads = n_heads
        self.n_chunks = n_chunks
        self.register_buffer(
            "fourier_freqs",
            torch.arange(1, n_freqs, 2).float() / (2 * n_freqs),
        )

        tw = torch.arange(width).float() / width
        th = torch.arange(height).float() / height
        th = self.encode_time(th.view([1, 1, -1, 1])).expand(1, -1, -1, width)
        tw = self.encode_time(tw.view([1, 1, 1, -1])).expand(1, -1, height, -1)
        self.register_buffer("pos_encoding", torch.cat([th, tw], dim=1))
        self.width = width
        self.height = height

    def forward(self, x):
        x_norm = self.norm(x)
        q = einops.rearrange(
            self.qconv_pos(self.pos_encoding)
            + self.qlin_input(x).view(x.size(0), -1, 1, 1),
            "b (heads c) h w -> b heads (h w) c",
            heads=self.n_heads,
        )
        k = einops.rearrange(
            self.klin(x_norm),
            "b (heads chunks c) -> b heads chunks c",
            heads=self.n_heads,
            chunks=self.n_chunks,
        )
        v = einops.rearrange(
            self.vlin(x_norm),
            "b (heads chunks c) -> b heads chunks c",
            heads=self.n_heads,
            chunks=self.n_chunks,
        )
        attn = F.scaled_dot_product_attention(q, k, v)
        return einops.rearrange(
            attn, "b heads (h w) c -> b (heads c) h w", w=self.width, h=self.height
        )

    def encode_time(self, t):
        return torch.cat(
            [
                torch.sin(t * self.fourier_freqs.view([1, -1, 1, 1]) * 2 * torch.pi),
                torch.cos(t * self.fourier_freqs.view([1, -1, 1, 1]) * 2 * torch.pi),
            ],
            dim=1,
        )


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
        blocks = [
            spectral_norm(nn.Conv2d(n_channels, hidden_dim, kernel_size=3, padding=1))
        ]
        for i in range(depth):
            for _ in range(n_resnet_blocks):
                blocks.append(
                    ResNet2d(
                        hidden_dim * 4**i,
                        total_depth=depth * n_resnet_blocks,
                    )
                )
            blocks.append(StackPool(2, 2))
            # blocks.append(
            #     ChannelAttention(
            #         hidden_dim * 2**i * 4,
            #         hidden_dim * 2**i * 2,
            #         n_heads=4,
            #         n_groups=N_GROUPS,
            #     )
            # )
            height //= 2
            width //= 2
        self.height = height
        self.width = width
        self.enc_net = nn.Sequential(*blocks)
        self.out = FlattenAttention(hidden_dim * 4**depth, latent_dim)

    def forward(self, x):
        y = self.enc_net(x)
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
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.inattn = UnFlattenAttention(
            latent_dim,
            hidden_dim * 4 ** (depth - 1),
            width=self.top_width * 2,
            height=self.top_height * 2,
        )
        blocks = []
        for i in reversed(range(depth)):
            if i < depth - 1:
                blocks.append(UnStackUpsample(2, 2))
            for _ in range(n_resnet_blocks):
                blocks.append(
                    ResNet2d(
                        hidden_dim * 4**i,
                        total_depth=depth * n_resnet_blocks,
                    )
                )
        self.dec_net = nn.Sequential(
            *blocks,
            nn.Conv2d(hidden_dim, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        y = self.inattn(x)
        # y = einops.rearrange(
        #     y,
        #     "... (c h w) -> ... c h w",
        #     w=self.top_width,
        #     h=self.top_height,
        #     c=self.hidden_dim * 4**self.depth,
        # )

        y = self.dec_net(y)
        log_var = torch.zeros_like(y)
        return NormalDistribution(y, 1e-2 * torch.exp(log_var))
