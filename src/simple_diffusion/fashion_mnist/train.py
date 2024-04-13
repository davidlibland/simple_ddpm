"""Train a diffusion model"""

import itertools

import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from simple_diffusion.fashion_mnist.plotting import sample_plotter
from simple_diffusion.model import DiffusionModel

# PyTorch TensorBoard support


SEED = 1337
NEPTUNE_PROJECT = "davidlibland/simplediffusion"
IMAGE_DIM = 8

# Set the seed:
L.seed_everything(SEED)


class DropLabels(Dataset):
    def __init__(self, dataset, data_shrink_factor=None):
        self.dataset = dataset
        self.data_shrink_factor = data_shrink_factor

    def __len__(self):
        if self.data_shrink_factor is not None:
            return len(self.dataset) // self.data_shrink_factor
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x


class CachedDataset(Dataset):
    """Move the dataset to the device and cache it."""

    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        self.cache = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.cache is None:
            self.cache = torch.stack([x.to(self.device) for x in self.dataset], dim=0)
        return self.cache[idx]


def train(
    batch_size=2**11,
    n_epochs=500,
    n_steps=100,
    check_val_every_n_epoch=100,
    beta=0.3,
    log_to_neptune=True,
    learning_rate=3e-2,
    beta_schedule_form="geometric",
    debug=False,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Create datasets for training & validation, download if necessary
    train_dataset = CachedDataset(
        DropLabels(
            torchvision.datasets.FashionMNIST(
                "./data", train=True, transform=transform, download=True
            ),
            data_shrink_factor=1000 if debug else None,
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    val_dataset = CachedDataset(
        DropLabels(
            torchvision.datasets.FashionMNIST(
                "./data", train=False, transform=transform, download=True
            ),
            data_shrink_factor=100 if debug else None,
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup the model:
    if beta_schedule_form == "geometric":
        beta_schedule = beta * ((1 - beta) ** (n_steps - torch.arange(n_steps)))
    elif beta_schedule_form == "linear":
        beta_start = 1e-4
        beta_end = beta
        beta_schedule = torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float)
    metrics = {
        # "fid": FrechetInceptionDistance(normalize=True, feature=64),
        # "kid": KernelInceptionDistance(
        #     normalize=True, subset_size=(batch_size // 2), feature=64
        # ),
    }
    model = DiffusionModel(
        beta_schedule=beta_schedule,
        latent_shape=(1, IMAGE_DIM, IMAGE_DIM),
        learning_rate=learning_rate,
        sample_plotter=sample_plotter,
        sample_metrics=None,  # metrics,
        sample_metric_pre_process_fn=lambda gray_img: gray_img.repeat(1, 3, 1, 1).to(
            "cpu"
        ),
        type="unet",
        n_steps=3,
        n_channels=1,
    )

    # Setup the logger and the trainer:
    # neptune_logger = NeptuneLogger(
    #     # api_key=NEPTUNE_API_TOKEN,  # replace with your own
    #     project=NEPTUNE_PROJECT,  # format "workspace-name/project-name"
    #     tags=["training", "diffusion", "gaussian_mixture"],  # optional
    #     mode="async" if log_to_neptune else "debug",
    # )
    logger = TensorBoardLogger("tb_logs", name="simplediffusion")
    logger.log_hyperparams(
        {
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "n_steps": n_steps,
            "beta_start": beta_schedule[0].item(),
            "beta_end": beta_schedule[-1].item(),
            "beta": beta,
            "learning_rate": learning_rate,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "image_dim": IMAGE_DIM,
            "seed": SEED,
            "beta_schedule_form": beta_schedule_form,
        }
    )
    trainer = L.Trainer(
        max_epochs=n_epochs,
        logger=logger,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
    trainer.fit(model, train_loader, val_loader)

    train_samples = torch.cat(list(itertools.islice(train_dataset, 0, 30)), dim=0)
    true_samples = train_samples.detach().cpu()
    fake_samples = trainer.model.generate(len(true_samples), seed=SEED).detach().cpu()
    fig = sample_plotter(
        real=true_samples,
        fake=fake_samples,
    )
    if hasattr(logger, "experiment"):
        logger.experiment.add_image("samples", fig)
    elif hasattr(logger, "run"):
        logger.run["samples"].upload(fig)


if __name__ == "__main__":
    train(beta_schedule_form="linear", beta=0.02)
