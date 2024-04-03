import matplotlib.pyplot as plt
import torch


def sample_plotter(real: torch.Tensor, fake: torch.Tensor) -> plt.Figure:
    """Plots 1-d histograms of real and fake samples."""
    fig, ax = plt.subplots()
    ax.hist(
        real.detach().cpu().numpy().flatten(),
        bins=100,
        alpha=0.5,
        label="True",
        color="blue",
    )
    ax.hist(
        fake.detach().cpu().numpy().flatten(),
        bins=100,
        alpha=0.5,
        label="Fake",
        color="red",
    )
    ax.legend()
    return fig
