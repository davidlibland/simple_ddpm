"""Train a diffusion model"""

import itertools

import lightning as L
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from lightning.pytorch.loggers import TensorBoardLogger
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
    batch_size=256,
    n_epochs=500,
    n_steps=100,
    check_val_every_n_epoch=10,
    beta=0.02,
    log_to_neptune=True,
    learning_rate=3e-2,
    beta_schedule_form="geometric",
    debug=False,
    cache=False,
):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Create datasets for training & validation, download if necessary
    train_dataset = DropLabels(
        torchvision.datasets.FashionMNIST(
            "./data", train=True, transform=transform, download=True
        ),
        data_shrink_factor=1000 if debug else None,
    )
    val_dataset = DropLabels(
        torchvision.datasets.FashionMNIST(
            "./data", train=False, transform=transform, download=True
        ),
        data_shrink_factor=100 if debug else None,
    )
    if cache and torch.cuda.is_available():
        train_dataset = CachedDataset(train_dataset, device="cuda")
        val_dataset = CachedDataset(val_dataset, device="cuda")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup the model:
    if beta_schedule_form == "geometric":
        raise NotImplementedError("Geometric schedule not implemented")
        beta_schedule = beta * ((1 - beta) ** (n_steps - torch.arange(n_steps)))
    elif beta_schedule_form == "linear":
        beta_start = 1e-4
        beta_end = beta
        diffusion_schedule_kwargs = {
            "schedule_type": beta_schedule_form,
            "beta_min": beta_start,
            "beta_max": beta_end,
            "n_steps": n_steps,
        }
    elif beta_schedule_form == "logit_linear":
        log_snr_min = -6
        log_snr_max = 6
        diffusion_schedule_kwargs = {
            "schedule_type": beta_schedule_form,
            "log_snr_min": log_snr_min,
            "log_snr_max": log_snr_max,
        }
    metrics = {
        # "fid": FrechetInceptionDistance(normalize=True, feature=64),
        # "kid": KernelInceptionDistance(
        #     normalize=True, subset_size=(batch_size // 2), feature=64
        # ),
    }
    model = DiffusionModel(
        latent_shape=(1, IMAGE_DIM, IMAGE_DIM),
        learning_rate=learning_rate,
        sample_plotter=sample_plotter,
        sample_metrics=None,  # metrics,
        sample_metric_pre_process_fn=lambda gray_img: gray_img.repeat(1, 3, 1, 1).to(
            "cpu"
        ),
        type="unet",
        u_steps=2,
        step_depth=1,
        n_channels=1,
        diffusion_schedule_kwargs=diffusion_schedule_kwargs,
        n_time_steps=n_steps,
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
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)

    train_samples = torch.stack(list(itertools.islice(train_dataset, 0, 30)), dim=0)
    true_samples = train_samples.detach().cpu()
    fake_samples = trainer.model.generate(len(true_samples), seed=SEED).detach().cpu()
    fig = sample_plotter(
        real=true_samples,
        fake=fake_samples,
    )
    if hasattr(logger, "experiment"):
        logger.experiment.add_figure("samples", fig)
    elif hasattr(logger, "run"):
        logger.run["samples"].upload(fig)


if __name__ == "__main__":
    train(beta_schedule_form="logit_linear", beta=0.02, debug=False)
