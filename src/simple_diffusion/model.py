"""A simple diffusion model"""

from typing import Dict, Callable

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import Metric

from simple_diffusion.fully_connected_denoiser import Denoiser as FC_Denoiser
from simple_diffusion.metrics import energy_coefficient
from simple_diffusion.unet import UNet


class DiffusionModel(L.LightningModule):
    def __init__(
        self,
        beta_schedule,
        sample_plotter=None,
        learning_rate=1e-1,
        sample_metrics: Dict[str, Metric] = None,
        sample_metric_pre_process_fn: Callable = None,
        **denoiser_kwargs,
    ):
        """
        A simple diffusion model.
        Args:
            beta_schedule (torch.Tensor): The schedule of beta values.
            sample_plotter (callable): A function that plots samples, returns a figure.
            learning_rate (float): The learning rate for the optimizer.
            sample_metrics (dict): A list of metrics to log, each should be a TorchMetric, which has
                an `update` method of signature `update(self, samples: Tensor, real: bool)`.
            sample_metric_pre_process_fn (callable): A function that preprocesses samples before
                passing them to the metrics.
            denoiser_kwargs (dict): The keyword arguments for the denoiser.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["sample_plotter", "metrics", "sample_metric_pre_process_fn"]
        )
        self.register_buffer("beta_schedule", beta_schedule)  # (T,)
        # assert self.beta_schedule[0] == 0
        alpha_schedule = torch.cumprod(1 - beta_schedule, dim=0)  # (T,)
        self.register_buffer("alpha_schedule", alpha_schedule)
        self.denoiser = self._build_denoiser(**denoiser_kwargs)
        self.latent_shape = denoiser_kwargs["latent_shape"]
        self.sample_plotter = sample_plotter
        self.sample_metrics = sample_metrics
        self.sample_metric_pre_process_fn = sample_metric_pre_process_fn

    def _build_denoiser(self, **denoiser_kwargs):
        """Build the denoiser network."""
        denoiser_kwargs = {**denoiser_kwargs}
        denoiser_type = denoiser_kwargs.pop("type", "fully_connected")
        if denoiser_type == "fully_connected":
            return FC_Denoiser(
                time_scale=len(self.beta_schedule) - 1,
                **denoiser_kwargs,
            )
        elif denoiser_type == "unet":
            return UNet(
                **denoiser_kwargs,
            )

    def decoder(self, z, t):
        """The decoder network, defined in terms of the denoiser."""
        beta = self.beta_schedule[t]
        alpha = self.alpha_schedule[t]
        noise_est = self.denoiser(z, t.expand(z.shape[0], *t.shape[1:]))
        if 1 - alpha == 0:
            factor = beta / torch.sqrt(torch.sum(beta[:t]))
        else:
            factor = beta / torch.sqrt(1 - alpha)
        return (z - factor * noise_est) / torch.sqrt(1 - beta)

    def training_step(self, batch):
        """The training step for the diffusion model."""
        loss, t = self._shared_step(batch)
        loss = loss.mean()
        self.logger.log_metrics({"train/loss": loss})
        return loss

    def _shared_step(self, batch):
        """The shared step for the training and validation steps."""
        x = batch
        n = x.shape[0]
        t = torch.randint(1, len(self.beta_schedule), (n,)).to(x.device)
        while len(t.shape) < len(x.shape):
            t = t.unsqueeze(-1)
        assert (t != 0).all()
        eps = torch.randn_like(x)
        alpha = self.alpha_schedule[t]
        z = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * eps
        eps_tilde = self.denoiser(z, t)
        loss = F.mse_loss(eps_tilde, eps, reduction="none")
        return loss, t

    def validation_step(self, batch, batch_idx):
        """The validation step for the diffusion model."""
        # Have the same seed as the batch, to keep
        # sample generation smooth across epochs:
        samples = self.generate(len(batch), seed=batch_idx)
        err = abs(samples.mean() - batch.mean())
        self.logger.log_metrics({"val/mean_err": err})
        e_coeff = energy_coefficient(samples, batch)
        self.logger.log_metrics({"val/energy_coeff": e_coeff})
        if batch_idx == 0 and self.sample_plotter is not None:
            fig = self.sample_plotter(batch, samples)
            self.logger.run["val/samples"].append(fig)

            # Add a scatter plot of losses at each time step:
            import matplotlib.pyplot as plt

            loss, t = self._shared_step(batch)
            loss = loss.view(loss.size(0), -1).mean(1)

            fig, ax = plt.subplots()
            ax.scatter(
                x=t.detach().cpu().numpy().flatten(),
                y=loss.detach().cpu().numpy().flatten(),
            )
            ax.set_xlabel("Diffusion Step")
            ax.set_ylabel("MSE Loss")
            # add a tick at the last time step:
            plt.text(
                1,
                0.5,
                "noise",
                horizontalalignment="right",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
                color="black",
            )
            plt.text(
                0,
                0.5,
                "image",
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
                color="black",
            )
            ax.set_title("Losses by Diffusion Step")
            self.logger.run["val/losses_by_time"].append(fig)

        if self.sample_metrics is not None:
            import timeit

            # Time the metric update:
            for k, metric in self.sample_metrics.items():
                start = timeit.default_timer()
                metric.update(self.sample_metric_pre_process_fn(samples), real=False)
                metric.update(self.sample_metric_pre_process_fn(batch), real=True)
                end = timeit.default_timer()
                self.logger.log_metrics({f"update_time/{k}": end - start})
        return err

    def on_validation_epoch_end(self):
        """Log the metrics at the end of the validation epoch."""
        if self.sample_metrics is not None:
            import timeit

            for k, metric in self.sample_metrics.items():

                start = timeit.default_timer()
                output = metric.compute()
                if isinstance(output, tuple):
                    self.logger.log_metrics(
                        {f"{k}_mean": output[0], f"{k}_std": output[1]}
                    )
                else:
                    self.logger.log_metrics({k: output})
                metric.reset()
                end = timeit.default_timer()
                self.logger.log_metrics({f"compute_time/{k}": end - start})

    def configure_optimizers(self):
        """The optimizer for the diffusion model."""
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def generate(self, n, seed=None):
        """Generate samples from the diffusion model."""
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
        else:
            gen = None
        z = torch.randn(n, *self.latent_shape, generator=gen).to(self.device)
        for t in range(len(self.beta_schedule) - 1, 0, -1):
            t = torch.tensor(t).to(self.device)
            while len(t.shape) < len(z.shape):
                t = t.unsqueeze(-1)
            beta = self.beta_schedule[t]
            mean = self.decoder(z, t)
            eps = torch.randn(*z.shape, generator=gen).to(self.device)
            z = mean + torch.sqrt(beta) * eps
        return mean
