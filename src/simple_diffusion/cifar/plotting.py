from typing import Callable

import matplotlib.pyplot as plt
import torch


def get_sample_plotter(
    image_inv_transform: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[[torch.Tensor, torch.Tensor], plt.Figure]:
    def sample_plotter(real: torch.Tensor, fake: torch.Tensor) -> plt.Figure:
        """Plots a 5x6 array of real and fake images the first three colums
        are real images, the last three columns are fake images."""
        fig, ax = plt.subplots(5, 6, figsize=(12, 10))
        real = image_inv_transform(real)
        fake = image_inv_transform(fake)

        for i in range(5):
            for j in range(3):
                # Plot 3-channel color images:

                ax[i, j].imshow(
                    real[i * 3 + j].permute(1, 2, 0).detach().cpu().numpy(),
                )
                ax[i, j].axis("off")
                ax[i, j].set_title("Real")
                ax[i, j + 3].imshow(
                    fake[i * 3 + j].permute(1, 2, 0).detach().cpu().numpy(),
                )
                ax[i, j + 3].axis("off")
                ax[i, j + 3].set_title("Fake")
        return fig

    return sample_plotter
