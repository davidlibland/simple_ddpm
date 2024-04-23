"""A unet for image denoising."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

N_GROUPS = 8


def _get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {activation}")


class Normalization(nn.Module):
    def __init__(self, num_groups, num_channels, time_embed_dim, activation="gelu"):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=num_channels, affine=False
        )

        # self.tnorm = nn.LayerNorm(time_embed_dim)
        # self.tactivation = _get_activation(activation)
        self.tdense_loc = nn.Linear(time_embed_dim, num_channels)
        self.tdense_scale = nn.Linear(time_embed_dim, num_channels)
        # Initialize these to zero:
        self.tdense_loc.weight.data.zero_()
        self.tdense_loc.bias.data.zero_()
        self.tdense_scale.weight.data.zero_()
        self.tdense_scale.bias.data.zero_()

    def forward(self, x, t):
        # t = self.tnorm(t)
        # t = self.tactivation(t)
        loc = self.tdense_loc(t)[:, :, None, None]
        scale = 1 + self.tdense_scale(t)[:, :, None, None]
        return self.norm(x) * scale + loc


class ResNetBlock(nn.Module):
    def __init__(
        self,
        n_channels,
        time_embed_dim,
        activation="gelu",
        time_norm=False,
        dropout=0.1,
        n_groups=N_GROUPS,
    ):
        super().__init__()
        self.time_norm = time_norm
        if self.time_norm:
            self.norm1 = Normalization(
                num_groups=n_groups,
                num_channels=n_channels,
                time_embed_dim=time_embed_dim,
                activation=activation,
            )
        else:
            self.norm1 = nn.GroupNorm(
                num_groups=n_groups,
                num_channels=n_channels,
            )
        self.activation1 = _get_activation(activation)
        self.conv1 = nn.Conv2d(n_channels, 2 * n_channels, kernel_size=3, padding=1)

        self.tdense = nn.Linear(time_embed_dim, 2 * n_channels)

        if self.time_norm:
            self.norm2 = Normalization(
                num_groups=N_GROUPS,
                num_channels=2 * n_channels,
                time_embed_dim=time_embed_dim,
                activation=activation,
            )
        else:
            self.norm2 = nn.GroupNorm(num_groups=n_groups, num_channels=2 * n_channels)
        self.activation2 = _get_activation(activation)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(2 * n_channels, n_channels, kernel_size=3, padding=1)

    def __repr__(self):
        return f"ResNetBlock(n_channels={self.conv1.in_channels})"

    def forward(self, x, t):
        # Expand the input:
        if self.time_norm:
            h = self.norm1(x, t)
        else:
            h = self.norm1(x)
        h = self.activation1(h)
        h = self.conv1(h)

        # (todo) Add time embedding:
        h = h + self.tdense(t)[:, :, None, None]

        # Collapse the input
        if self.time_norm:
            h = self.norm2(h, t)
        else:
            h = self.norm2(h)
        h = self.activation2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip Connection:
        h = h + x
        return h


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, n_heads=8, n_freqs=32, n_groups=N_GROUPS):
        super().__init__()
        if in_channels % n_heads != 0:
            raise ValueError(
                f"Number of heads {n_heads} must divide number of channels {in_channels}"
            )
        self.norm = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)
        self.conv = nn.Conv2d(in_channels + 2 * n_freqs, in_channels * 3, kernel_size=1)
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
            rearrange(
                t,
                "b (heads c) h w -> b heads (h w) c",
                heads=self.n_heads,
            )
            for t in qkv
        ]
        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = rearrange(
            attn, "b heads (h w) c -> b (heads c) h w", h=x.size(-2), w=x.size(-1)
        )
        return x + attn

    def encode_time(self, t):
        return torch.cat(
            [
                torch.sin(t * self.fourier_freqs.view([1, -1, 1, 1]) * 2 * torch.pi),
                torch.cos(t * self.fourier_freqs.view([1, -1, 1, 1]) * 2 * torch.pi),
            ],
            dim=1,
        )


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        activation="gelu",
        depth=1,
        attn=False,
        dropout=0.1,
        n_groups=N_GROUPS,
        n_heads=8,
    ):
        super().__init__()

        self.resnets = nn.ModuleList(
            [
                ResNetBlock(
                    in_channels,
                    time_embed_dim=time_embed_dim,
                    activation=activation,
                    dropout=dropout,
                    n_groups=n_groups,
                )
                for _ in range(depth)
            ]
        )
        self.attns = nn.ModuleList(
            [
                (
                    ChannelAttention(
                        in_channels, n_heads=n_heads, n_freqs=32, n_groups=n_groups
                    )
                    if attn
                    else nn.Identity()
                )
                for _ in range(depth)
            ]
        )

        self.convout = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def __repr__(self):
        return f"Up(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels})"

    def forward(self, x, skip, t):
        # Process the input:
        h = x
        for resnet, attn in zip(self.resnets, self.attns):
            h = resnet(h, t)
            h = attn(h)

        h = self.convout(h)

        # Upsample:
        h = self.upsample(h)

        # Add the skip connection:
        h = h + skip
        return h


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        activation="gelu",
        depth=1,
        attn=False,
        dropout=0.1,
        n_groups=N_GROUPS,
        n_heads=8,
    ):
        super().__init__()

        self.resnets = nn.ModuleList(
            [
                ResNetBlock(
                    in_channels,
                    time_embed_dim=time_embed_dim,
                    activation=activation,
                    dropout=dropout,
                    n_groups=n_groups,
                )
                for _ in range(depth)
            ]
        )
        self.attns = nn.ModuleList(
            [
                (
                    ChannelAttention(
                        in_channels, n_heads=n_heads, n_freqs=32, n_groups=n_groups
                    )
                    if attn
                    else nn.Identity()
                )
                for _ in range(depth)
            ]
        )

        self.convout = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)

    def __repr__(self):
        return f"Down(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels})"

    def forward(self, x, t):
        # Process the input:
        h = x
        for resnet, attn in zip(self.resnets, self.attns):
            h = resnet(h, t)
            h = attn(h)

        h = self.convout(h)

        # downsample:
        h = self.pool(h)
        return h


class UNet(nn.Module):
    def __init__(
        self,
        u_steps,
        step_depth,
        n_channels,
        activation="gelu",
        n_freqs=64,
        initial_hidden=8,
        mid_attn=True,
        attn_resolutions=(1,),
        dropout=0.1,
        n_groups=N_GROUPS,
        n_heads=8,
    ):
        super().__init__()
        self.attn = mid_attn
        channel_list = [initial_hidden * 2**i for i in range(u_steps)]
        self.register_buffer(
            "fourier_freqs",
            torch.arange(1, n_freqs, 2).float() / (2 * n_freqs),
        )

        # Input conv:
        self.input_norm = Normalization(
            num_groups=1,
            num_channels=n_channels,
            time_embed_dim=n_freqs,
            activation=activation,
        )
        self.input_conv = nn.Conv2d(
            n_channels, initial_hidden, kernel_size=3, padding=1
        )

        # Down steps
        self.downs = nn.ModuleList(
            [
                Down(
                    in_channels,
                    out_channels,
                    time_embed_dim=n_freqs,
                    activation=activation,
                    depth=step_depth,
                    attn=i in attn_resolutions,
                    dropout=dropout,
                    n_groups=n_groups,
                    n_heads=n_heads,
                )
                for i, (in_channels, out_channels) in enumerate(
                    zip(channel_list[:-1], channel_list[1:])
                )
            ]
        )

        # Middle Steps

        self.pre_middle = nn.ModuleList(
            [
                ResNetBlock(
                    channel_list[-1],
                    time_embed_dim=n_freqs,
                    activation=activation,
                    dropout=dropout,
                    n_groups=n_groups,
                )
                for _ in range(step_depth - 1)
            ]
        )
        self.post_middle = nn.ModuleList(
            [
                ResNetBlock(
                    channel_list[-1],
                    time_embed_dim=n_freqs,
                    activation=activation,
                    dropout=dropout,
                    n_groups=n_groups,
                )
                for _ in range(step_depth - 1)
            ]
        )
        if self.attn:
            self.middle = ChannelAttention(
                channel_list[-1], n_heads=n_heads, n_groups=n_groups
            )
        else:
            self.middle = nn.Identity()

        # Up steps
        self.ups = nn.ModuleList(
            [
                Up(
                    in_channels,
                    out_channels,
                    time_embed_dim=n_freqs,
                    activation=activation,
                    depth=step_depth,
                    attn=i in attn_resolutions,
                    dropout=dropout,
                    n_groups=n_groups,
                    n_heads=n_heads,
                )
                for i, (in_channels, out_channels) in reversed(
                    list(enumerate(zip(channel_list[1:], channel_list[:-1])))
                )
            ]
        )

        # Output conv:
        self.out_norm = Normalization(
            num_groups=n_groups,
            num_channels=initial_hidden,
            time_embed_dim=n_freqs,
            activation=activation,
        )
        self.out_activation = _get_activation(activation)
        self.output_conv = nn.Conv2d(
            initial_hidden, n_channels, kernel_size=3, padding=1
        )
        # Initalize the output conv to be zero:
        self.output_conv.weight.data.zero_()
        self.output_conv.bias.data.zero_()

    def forward(self, x, t):
        """
        Forward pass of the UNet.

        Args:
            x (torch.Tensor): The input tensor of shape (batch, channels, height, width).
            t (torch.Tensor): The time tensor of shape (batch, 1). It is assumed that
                time is scaled to be between 0 and 1.
        """
        # Embed the time:
        temb = self.encode_time(t.reshape(t.size(0), -1))
        skips = []
        x = self.input_norm(x, temb)
        x = self.input_conv(x)
        for down in self.downs:
            skips.append(x)
            x = down(x, temb)
        for resnet in self.pre_middle:
            x = resnet(x, temb)
        x = self.middle(x)
        for resnet in self.post_middle:
            x = resnet(x, temb)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, temb)
        x = self.out_norm(x, temb)
        x = self.out_activation(x)
        x = self.output_conv(x)
        return x

    def encode_time(self, t):
        return torch.cat(
            [
                torch.sin(t * self.fourier_freqs * 2 * torch.pi),
                torch.cos(t * self.fourier_freqs * 2 * torch.pi),
            ],
            dim=-1,
        )
