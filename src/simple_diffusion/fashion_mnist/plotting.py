import matplotlib.pyplot as plt
import torch


def sample_plotter(real: torch.Tensor, fake: torch.Tensor) -> plt.Figure:
    """Plots a 5x6 array of real and fake images the first three colums
    are real images, the last three columns are fake images."""
    fig, ax = plt.subplots(5, 6, figsize=(12, 10))
    for i in range(5):
        for j in range(3):
            ax[i, j].imshow(
                real[i * 3 + j].detach().cpu().numpy().squeeze(), cmap="gray"
            )
            ax[i, j].axis("off")
            ax[i, j].set_title("Real")
            ax[i, j + 3].imshow(
                fake[i * 3 + j].detach().cpu().numpy().squeeze(), cmap="gray"
            )
            ax[i, j + 3].axis("off")
            ax[i, j + 3].set_title("Fake")
    return fig
