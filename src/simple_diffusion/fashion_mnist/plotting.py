import matplotlib.pyplot as plt
import torch


def sample_plotter(real: torch.Tensor, fake: torch.Tensor) -> plt.Figure:
    """Plots 2-d scatter plot of real and fake samples."""
    fig, ax = plt.subplots()
    real = real.detach().cpu().numpy()
    ax.scatter(
        real[:, 0],
        real[:, 1],
        alpha=0.5,
        label="True",
        color="blue",
    )
    fake = fake.detach().cpu().numpy()
    ax.scatter(
        fake[:, 0],
        fake[:, 1],
        alpha=0.5,
        label="Fake",
        color="red",
    )
    ax.legend()
    return fig
