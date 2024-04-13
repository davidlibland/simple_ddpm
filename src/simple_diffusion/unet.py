"""A unet for image denoising."""

import torch
from torch import nn


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
    def __init__(self, n_channels, time_embed_dim, activation="gelu"):
        super().__init__()
        self.norm1 = Normalization(
            num_groups=1,
            num_channels=n_channels,
            time_embed_dim=time_embed_dim,
            activation=activation,
        )
        self.activation1 = _get_activation(activation)
        self.conv1 = nn.Conv2d(n_channels, 2 * n_channels, kernel_size=3, padding=1)

        self.norm2 = Normalization(
            num_groups=1,
            num_channels=2 * n_channels,
            time_embed_dim=time_embed_dim,
            activation=activation,
        )
        self.activation2 = _get_activation(activation)
        self.conv2 = nn.Conv2d(2 * n_channels, n_channels, kernel_size=3, padding=1)

    def __repr__(self):
        return f"ResNetBlock(n_channels={self.conv1.in_channels})"

    def forward(self, x, t):
        # Expand the input:
        h = self.norm1(x, t)
        h = self.activation1(h)
        h = self.conv1(h)

        # (todo) Add time embedding:

        # Collapse the input
        h = self.norm2(h, t)
        h = self.activation2(h)
        h = self.conv2(h)

        # Skip Connection:
        h = h + x
        return h


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, activation="gelu"):
        super().__init__()

        self.resnet = ResNetBlock(
            in_channels, time_embed_dim=time_embed_dim, activation=activation
        )

        self.convout = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def __repr__(self):
        return f"Up(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels})"

    def forward(self, x, skip, t):
        # Process the input:
        h = self.resnet(x, t)

        h = self.convout(h)

        # Upsample:
        h = self.upsample(h)

        # Add the skip connection:
        h = h + skip
        return h


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim, activation="gelu"):
        super().__init__()

        self.resnet = ResNetBlock(
            in_channels, time_embed_dim=time_embed_dim, activation=activation
        )

        self.convout = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)

    def __repr__(self):
        return f"Down(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels})"

    def forward(self, x, t):
        # Process the input:
        h = self.resnet(x, t)

        h = self.convout(h)

        # downsample:
        h = self.pool(h)
        return h


class UNet(nn.Module):
    def __init__(
        self,
        n_steps,
        n_channels,
        time_scale,
        activation="gelu",
        n_freqs=32,
        initial_hidden=8,
    ):
        super().__init__()
        channel_list = [initial_hidden * 2**i for i in range(n_steps)]
        self.register_buffer(
            "fourier_freqs",
            torch.arange(1, n_freqs, 2).float() / (2 * n_freqs) / time_scale,
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
                )
                for in_channels, out_channels in zip(
                    channel_list[:-1], channel_list[1:]
                )
            ]
        )

        # Middle Steps
        self.middle = nn.Identity()

        # Up steps
        self.ups = nn.ModuleList(
            [
                Up(
                    in_channels,
                    out_channels,
                    time_embed_dim=n_freqs,
                    activation=activation,
                )
                for in_channels, out_channels in reversed(
                    list(zip(channel_list[1:], channel_list[:-1]))
                )
            ]
        )

        # Output conv:
        self.out_norm = Normalization(
            num_groups=1,
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
        # Embed the time:
        temb = self.encode_time(t.reshape(t.size(0), -1))
        skips = []
        x = self.input_norm(x, temb)
        x = self.input_conv(x)
        for down in self.downs:
            skips.append(x)
            x = down(x, temb)
        x = self.middle(x)
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
