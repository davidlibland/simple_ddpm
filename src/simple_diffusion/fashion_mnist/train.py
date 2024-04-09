"""Train a diffusion model"""

import itertools

import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.loggers import NeptuneLogger
from torch.utils.data import Dataset, DataLoader

from simple_diffusion.fashion_mnist.plotting import sample_plotter
from simple_diffusion.model import DiffusionModel

# PyTorch TensorBoard support


SEED = 1337
NEPTUNE_PROJECT = "davidlibland/simplediffusion"
IMAGE_DIM = 28

# Set the seed:
L.seed_everything(SEED)


class DropLabels(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x


def train(
    n_samples=1000,
    batch_size=1000,
    n_epochs=10000,
    n_steps=100,
    beta=0.3,
    log_to_neptune=True,
    learning_rate=3e-2,
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Create datasets for training & validation, download if necessary
    train_dataset = DropLabels(
        torchvision.datasets.FashionMNIST(
            "./data", train=True, transform=transform, download=True
        )
    )
    val_dataset = DropLabels(
        torchvision.datasets.FashionMNIST(
            "./data", train=False, transform=transform, download=True
        )
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Setup the model:
    beta_schedule = beta * ((1 - beta) ** (n_steps - torch.arange(n_steps)))
    model = DiffusionModel(
        beta_schedule=beta_schedule,
        latent_shape=(1, IMAGE_DIM, IMAGE_DIM),
        learning_rate=learning_rate,
        sample_plotter=sample_plotter,
    )

    # Setup the logger and the trainer:
    neptune_logger = NeptuneLogger(
        # api_key=NEPTUNE_API_TOKEN,  # replace with your own
        project=NEPTUNE_PROJECT,  # format "workspace-name/project-name"
        tags=["training", "diffusion", "gaussian_mixture"],  # optional
        mode="async" if log_to_neptune else "debug",
    )
    neptune_logger.run["training_params"] = {
        "n_samples": n_samples,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "beta": beta,
        "learning_rate": learning_rate,
    }
    trainer = L.Trainer(
        max_epochs=n_epochs,
        logger=neptune_logger,
        check_val_every_n_epoch=100,
    )
    trainer.fit(model, train_loader, val_loader)

    train_samples = torch.cat(list(itertools.islice(train_dataset, 0, 30)), dim=0)
    true_samples = train_samples.detach().cpu()
    fake_samples = trainer.model.generate(len(true_samples), seed=SEED).detach().cpu()
    fig = sample_plotter(
        real=true_samples,
        fake=fake_samples,
    )

    neptune_logger.run["samples"].upload(fig)


if __name__ == "__main__":
    train()
