import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        time_embed_dim,
        dropout=0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.activation1 = nn.GELU()
        self.conv1 = nn.Linear(latent_dim, 2 * latent_dim)

        self.tdense = nn.Linear(time_embed_dim, 2 * latent_dim)

        self.norm2 = nn.LayerNorm(2 * latent_dim)
        self.activation2 = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Linear(2 * latent_dim, latent_dim)
        # Initialize the final weights to zero:
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def __repr__(self):
        return f"ResNetBlock(n_channels={self.conv1.in_channels})"

    def forward(self, x, t):
        # Expand the input:
        h = self.norm1(x)
        h = self.activation1(h)
        h = self.conv1(h)

        h = h + self.tdense(t)

        # Collapse the input
        h = self.norm2(h)
        h = self.activation2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip Connection:
        h = h + x
        return h


class ResNet(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_blocks=3,
        dropout=0.1,
        n_freqs=32,
    ):
        super().__init__()
        self.register_buffer(
            "fourier_freqs",
            torch.arange(1, n_freqs, 2).float() / (2 * n_freqs),
        )
        self.blocks = nn.ModuleList(
            [ResNetBlock(latent_dim, n_freqs, dropout) for _ in range(n_blocks)]
        )

    def forward(self, x, t):
        t_emb = self.encode_time(t)
        for block in self.blocks:
            x = block(x, t_emb)
        return x

    def encode_time(self, t):
        return torch.cat(
            [
                torch.sin(t * self.fourier_freqs * 2 * torch.pi),
                torch.cos(t * self.fourier_freqs * 2 * torch.pi),
            ],
            dim=-1,
        )
