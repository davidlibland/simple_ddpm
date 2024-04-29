"""Train a diffusion model"""

import itertools

import lightning as L
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
from torch.utils.data import Dataset, DataLoader

from simple_diffusion.cifar.plotting import get_sample_plotter, get_noisy_images_plotter
from simple_diffusion.vdm_model import DiffusionModel

# PyTorch TensorBoard support


SEED = 1337
NEPTUNE_PROJECT = "davidlibland/simplediffusion"
IMAGE_DIM = 32

# Set the seed:
L.seed_everything(SEED)
torch.set_float32_matmul_precision("medium")


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


class LitDataModule(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # or
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)


def log_figure(name, figure, logger):
    """Log a figure to the logger."""
    if hasattr(logger, "experiment"):
        logger.experiment.add_figure(name, figure)
    elif hasattr(logger, "run"):
        logger.run[name].upload(figure)


def train(
    batch_size=512,  # 2**11,
    n_epochs=500,
    n_steps=1000,
    check_val_every_n_epoch=50,
    learning_rate=3e-5,
    u_steps=3,
    step_depth=2,
    initial_hidden=128,
    dropout=0.1,
    beta_schedule_form="linear",
    debug=False,
    cache=True,
    find_batch_size=False,
    find_lr=False,
    n_groups=8,
    n_heads=8,
):
    # Compute the mean and std of the cifar channels:
    mean = 0.5
    std = 0.5

    def image_inv_transform(img):
        return img.clip(min=-1, max=1) * std + mean

    sample_plotter = get_sample_plotter(image_inv_transform)
    noisy_image_plotter = get_noisy_images_plotter(image_inv_transform)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
            transforms.Normalize((mean, mean, mean), (std, std, std)),
        ]
    )

    # Create datasets for training & validation, download if necessary
    train_dataset = DropLabels(
        torchvision.datasets.CIFAR10(
            "./data", train=True, transform=transform, download=True
        ),
        data_shrink_factor=1000 if debug else None,
    )
    val_dataset = DropLabels(
        torchvision.datasets.CIFAR10(
            "./data", train=False, transform=transform, download=True
        ),
        data_shrink_factor=100 if debug else None,
    )
    if cache and torch.cuda.is_available():
        train_dataset = CachedDataset(train_dataset, device="cuda")
        val_dataset = CachedDataset(val_dataset, device="cuda")
    # Check that images are normalized:
    assert train_dataset[0].min() >= -1
    assert train_dataset[0].max() <= 1

    datamodule = LitDataModule(train_dataset, val_dataset, batch_size=batch_size)

    # Setup the model:
    if beta_schedule_form == "geometric":
        raise NotImplementedError("Geometric schedule not implemented")
        beta_schedule = beta * ((1 - beta) ** (n_steps - torch.arange(n_steps)))
    elif beta_schedule_form == "linear":
        diffusion_schedule_kwargs = {
            "schedule_type": beta_schedule_form,
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
        latent_shape=(3, IMAGE_DIM, IMAGE_DIM),
        learning_rate=learning_rate,
        sample_plotter=sample_plotter,
        sample_metrics=None,  # metrics,
        sample_metric_pre_process_fn=lambda img: img.to("cpu"),
        type="unet",
        u_steps=u_steps,
        step_depth=step_depth,
        initial_hidden=initial_hidden,
        n_channels=3,
        diffusion_schedule_kwargs=diffusion_schedule_kwargs,
        noisy_image_plotter=noisy_image_plotter,
        mid_attn=True,
        attn_resolutions=(1,),
        dropout=dropout,
        n_groups=n_groups,
        n_heads=n_heads,
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
            "learning_rate": learning_rate,
            "check_val_every_n_epoch": check_val_every_n_epoch,
            "image_dim": IMAGE_DIM,
            "seed": SEED,
            **diffusion_schedule_kwargs,
        }
    )

    trainer = L.Trainer(
        max_epochs=n_epochs,
        logger=logger,
        check_val_every_n_epoch=check_val_every_n_epoch,
        gradient_clip_val=1.0,
    )
    tuner = Tuner(trainer)
    if find_batch_size:
        tuner.scale_batch_size(model, mode="binsearch", datamodule=datamodule)

    if find_lr:
        # Run learning rate finder
        lr_finder = tuner.lr_find(model, datamodule=datamodule)

        # Results can be found in
        print(lr_finder.results)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        model.hparams.learning_rate = new_lr
    trainer.fit(model, datamodule=datamodule)

    train_samples = torch.stack(list(itertools.islice(train_dataset, 0, 60)), dim=0)
    true_samples = train_samples.detach().cpu()
    fake_samples = trainer.model.generate(len(true_samples), seed=SEED).detach().cpu()
    fig = sample_plotter(
        real=true_samples,
        fake=fake_samples,
    )
    log_figure("samples", fig, logger)


if __name__ == "__main__":
    train(beta_schedule_form="logit_linear", debug=False, cache=False)
