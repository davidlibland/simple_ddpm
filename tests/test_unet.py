import torch

from simple_diffusion.unet import ChannelAttention


def test_channel_attention():
    """A smoke test for the channel attention layer."""
    ca = ChannelAttention(in_channels=10, n_heads=2)
    x = torch.randn(2, 10, 32, 32)
    y = ca(x)
    assert y.shape == x.shape
