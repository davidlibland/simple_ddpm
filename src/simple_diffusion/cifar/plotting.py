from typing import Callable, List

import matplotlib.pyplot as plt
import torch


def get_sample_plotter(
    image_inv_transform: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[[torch.Tensor, torch.Tensor], plt.Figure]:
    def sample_plotter(real: torch.Tensor, fake: torch.Tensor) -> plt.Figure:
        """Plots a 5x6 array of real and fake images the first three colums
        are real images, the last three columns are fake images."""
        nrows = 10
        ncols = 6
        fig, ax = plt.subplots(nrows, 2 * ncols, figsize=(4 * ncols, 2 * nrows))
        real = image_inv_transform(real)
        fake = image_inv_transform(fake)
        ax = ax.reshape(nrows, 2 * ncols)

        for i in range(nrows):
            for j in range(ncols):
                # Plot 3-channel color images:

                ax[i, j].imshow(
                    real[i * ncols + j].permute(1, 2, 0).detach().cpu().numpy(),
                )
                ax[i, j].axis("off")
                ax[i, j].set_title("Real")
                ax[i, j + ncols].imshow(
                    fake[i * ncols + j].permute(1, 2, 0).detach().cpu().numpy(),
                )
                ax[i, j + ncols].axis("off")
                ax[i, j + ncols].set_title("Fake")
        return fig

    return sample_plotter


def get_noisy_images_plotter(
    image_inv_transform: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[[torch.Tensor, torch.Tensor], plt.Figure]:

    def plot_noisy_images(image_list: List[torch.Tensor]) -> plt.Figure:
        """Plots an array of images."""
        n_cols = len(image_list)
        n_rows = image_list[0].shape[0]
        image_list = [image_inv_transform(images) for images in image_list]
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        ax = ax.reshape(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                ax[i, j].imshow(
                    image_list[j][i].permute(1, 2, 0).detach().cpu().numpy()
                )
                ax[i, j].axis("off")

        return fig

    return plot_noisy_images
