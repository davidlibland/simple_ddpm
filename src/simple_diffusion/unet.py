"""A unet for image denoising."""

from torch import nn


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def __repr__(self):
        return f"Up(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels})"

    def forward(self, x, skip, t):
        # Ignore t for now.
        h = self.conv(x)
        h = self.upsample(h)
        h = h + skip
        return h


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)

    def __repr__(self):
        return f"Down(in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels})"

    def forward(self, x, t):
        # Ignore t for now.
        h = self.conv(x)
        h = self.pool(h)
        return h


class UNet(nn.Module):
    def __init__(self, n_steps, n_channels):
        super().__init__()
        channel_list = [n_channels * 2**i for i in range(n_steps)]
        # Down steps
        self.downs = nn.ModuleList(
            [
                Down(in_channels, out_channels)
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
                Up(in_channels, out_channels)
                for in_channels, out_channels in reversed(
                    list(zip(channel_list[1:], channel_list[:-1]))
                )
            ]
        )

    def forward(self, x, t):
        skips = []
        for down in self.downs:
            skips.append(x)
            x = down(x, t)
        x = self.middle(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t)
        return x
