"""Train a diffusion model"""

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from simple_diffusion.gaussian_1d.plotting import sample_plotter
from simple_diffusion.ldm_model import LatentDiffusionModel

SEED = 1337
NEPTUNE_PROJECT = "davidlibland/simplediffusion"

# Set the seed:
L.seed_everything(SEED)


# Setup the dataset:
class GaussianMixture(Dataset):
    def __init__(self, means, stds, n_samples=1000):
        """A multimodal dataset"""
        self.means = means
        self.stds = stds
        self.n_modes = len(means)
        self.n_samples = n_samples
        self.samples = torch.cat(
            [
                means[i] + stds[i] * torch.randn(n_samples // self.n_modes, 1)
                for i in range(self.n_modes)
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def log_figure(name, figure, logger):
    """Log a figure to the logger."""
    if hasattr(logger, "experiment"):
        logger.experiment.add_figure(name, figure)
    elif hasattr(logger, "run"):
        logger.run[name].upload(figure)


def train(
    means=(-1, 1),
    stds=(0.1, 0.1),
    n_samples=1000,
    batch_size=1000,
    n_epochs=3000,
    n_steps=100,
    beta=0.3,
    log_to_neptune=True,
):
    train_dataset = GaussianMixture(means, stds, n_samples=n_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = GaussianMixture(means, stds, n_samples=n_samples)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Setup the model:
    model = LatentDiffusionModel(
        sample_plotter=sample_plotter,
        diffusion_schedule_kwargs={
            "schedule_type": "logit_linear",
            "log_snr_max": 6,
            "log_snr_min": -6,
        },
        latent_dim=128,
        denoiser_kwargs={"type": "fully_connected"},
        encoder_kwargs={"type": "fully_connected", "data_dim": 1},
        decoder_kwargs={"type": "fully_connected", "data_dim": 1},
        n_time_steps=30,
        learning_rate=1e-3,
    )

    # Setup the logger and the trainer:
    # neptune_logger = NeptuneLogger(
    #     # api_key=NEPTUNE_API_TOKEN,  # replace with your own
    #     project=NEPTUNE_PROJECT,  # format "workspace-name/project-name"
    #     tags=["training", "diffusion", "gaussian_mixture"],  # optional
    #     mode="async" if log_to_neptune else "debug",
    # )
    logger = TensorBoardLogger("tb_logs", name="simplediffusion")
    trainer = L.Trainer(
        max_epochs=n_epochs,
        logger=logger,
        check_val_every_n_epoch=100,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)

    true_samples = train_dataset.samples.detach().cpu().numpy()
    fig = sample_plotter(
        real=train_dataset.samples.detach().cpu(),
        fake=trainer.model.generate(len(true_samples), seed=SEED).detach().cpu(),
    )

    log_figure("samples", fig, logger)


if __name__ == "__main__":
    train()