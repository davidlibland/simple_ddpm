"""A simple diffusion model"""

from typing import Dict, Callable, Tuple

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

from simple_diffusion.fully_connected_denoiser import Denoiser as FC_Denoiser
from simple_diffusion.metrics import energy_coefficient
from simple_diffusion.unet import UNet


class LinearDiffusionSchedule(nn.Module):
    def __init__(self, beta_min=1e-2, beta_max=0.99, n_steps=100):
        super().__init__()

        beta_schedule = torch.linspace(beta_min, beta_max, n_steps, dtype=torch.float)
        alpha_schedule = torch.cumprod(1 - beta_schedule, dim=0)  # (T,)
        self.register_buffer("alpha_schedule", alpha_schedule)
        self.register_buffer("beta_schedule", beta_schedule)
        signal = torch.sqrt(self.alpha_schedule)
        noise = 1 - self.alpha_schedule
        log_snr = torch.log(torch.square(signal) / noise)
        self.register_buffer("log_snr", log_snr)

    @property
    def n_steps(self):
        return len(self.alpha_schedule)

    def log_signal_to_noise(self):
        """Returns the log signal to noise ratio."""
        return self.log_snr

    def forward(self, t):
        """Returns the signal at time t"""
        scale_t_2 = torch.sigmoid(self.log_snr[t])
        var_t = 1 - scale_t_2
        scale_t = torch.sqrt(scale_t_2)
        scale_s_2 = torch.sigmoid(self.log_snr[t - 1])
        var_s = 1 - scale_s_2
        scale_s = torch.sqrt(scale_s_2)
        beta_t = 1 - scale_t_2 / scale_s_2
        scale_t_s = scale_t / scale_s
        return {
            "alpha": scale_t_2,
            "beta": beta_t,
            "1-alpha": var_t,
            "1-beta": scale_t_2 / scale_s_2,
        }


class LogitLinearSNR(nn.Module):
    def __init__(self, log_snr_min=-6, log_snr_max=6, n_steps=100):
        super().__init__()

        log_snr = torch.linspace(log_snr_max, log_snr_min, n_steps, dtype=torch.float)
        self.register_buffer("log_snr", log_snr)

    @property
    def n_steps(self):
        return len(self.log_snr)

    def log_signal_to_noise(self):
        """Returns the log signal to noise ratio."""
        return self.log_snr

    def forward(self, t):
        """Returns the signal at time t"""
        scale_t_2 = torch.sigmoid(self.log_snr[t])
        var_t = torch.sigmoid(-self.log_snr[t])
        beta_t = -torch.expm1(self.log_snr[t] - self.log_snr[t - 1]) * torch.sigmoid(
            -self.log_snr[t]
        )
        return {
            "alpha": scale_t_2,
            "beta": beta_t,
            "1-alpha": var_t,
            "1-beta": 1 - beta_t,
        }


class DiffusionModel(L.LightningModule):
    def __init__(
        self,
        latent_shape: Tuple[int],
        sample_plotter=None,
        learning_rate=1e-1,
        sample_metrics: Dict[str, Metric] = None,
        sample_metric_pre_process_fn: Callable = None,
        diffusion_schedule_kwargs: Dict = {},
        noisy_image_plotter=None,
        **denoiser_kwargs,
    ):
        """
        A simple diffusion model.
        Args:
            latent_shape (Tuple[int]): The shape of the latent space.
            sample_plotter (callable): A function that plots samples, returns a figure.
            learning_rate (float): The learning rate for the optimizer.
            sample_metrics (dict): A list of metrics to log, each should be a TorchMetric, which has
                an `update` method of signature `update(self, samples: Tensor, real: bool)`.
            sample_metric_pre_process_fn (callable): A function that preprocesses samples before
                passing them to the metrics.
            diffusion_schedule_kwargs (dict): The keyword arguments for the diffusion schedule.
            denoiser_kwargs (dict): The keyword arguments for the denoiser.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "sample_plotter",
                "metrics",
                "sample_metric_pre_process_fn",
                "noisy_image_plotter",
            ]
        )
        self.diffusion_schedule = self._build_diffusion_schedule(
            **diffusion_schedule_kwargs
        )
        self.denoiser = self._build_denoiser(**denoiser_kwargs)
        self.sample_plotter = sample_plotter
        self.sample_metrics = sample_metrics
        self.sample_metric_pre_process_fn = sample_metric_pre_process_fn
        self.noisy_image_plotter = noisy_image_plotter

    def log_signal_to_noise(self):
        """Compute the log signal to noise ratio of the denoiser."""
        t = torch.arange(0, self.diffusion_schedule.n_steps)
        return t, self.diffusion_schedule.log_signal_to_noise()

    def generative_variance_at_zero_mean(self):
        """Compute the variance at time 0."""
        var = 1
        for t in range(self.diffusion_schedule.n_steps - 1, 0, -1):
            t = torch.tensor(t).to(self.device)
            beta = self.diffusion_schedule(t)["beta"].detach().cpu().numpy()
            one_m_beta = self.diffusion_schedule(t)["1-beta"].detach().cpu().numpy()
            var = var / one_m_beta + beta
        return var

    def _build_denoiser(self, **denoiser_kwargs):
        """Build the denoiser network."""
        denoiser_kwargs = {**denoiser_kwargs}
        denoiser_type = denoiser_kwargs.pop("type", "fully_connected")
        if denoiser_type == "fully_connected":
            return FC_Denoiser(
                time_scale=self.diffusion_schedule.n_steps - 1,
                latent_shape=self.hparams.latent_shape,
                **denoiser_kwargs,
            )
        elif denoiser_type == "unet":
            return UNet(
                time_scale=self.diffusion_schedule.n_steps - 1,
                **denoiser_kwargs,
            )

    def _build_diffusion_schedule(self, schedule_type="linear", **kwargs):
        """Build the denoiser network."""
        if schedule_type == "linear":
            return LinearDiffusionSchedule(**kwargs)
        elif schedule_type == "logit_linear":
            return LogitLinearSNR(**kwargs)
        else:
            raise NotImplementedError(f"Schedule type {schedule_type} not implemented.")

    def decoder(self, z, t):
        """The decoder network, defined in terms of the denoiser."""
        schedule = self.diffusion_schedule(t)
        beta = schedule["beta"]
        noise_est = self.denoiser(z, t.expand(z.shape[0], *t.shape[1:]))
        factor = beta / torch.sqrt(schedule["1-alpha"])
        return (z - factor * noise_est) / torch.sqrt(schedule["1-beta"])

    def training_step(self, batch):
        """The training step for the diffusion model."""
        loss, t = self._shared_step(batch)
        loss = loss.mean()
        self.log("train/loss", loss, prog_bar=True)

        # Misc logging:
        for name, param in self.named_parameters():
            if "tdense" in name:
                self.log(f"param/{name}", param.norm())
        return loss

    def on_after_backward(self) -> None:
        """Log the gradient norm after the backward pass."""
        for name, param in self.named_parameters():
            self.log(f"grad/{name}", param.grad.norm())

    def on_train_epoch_start(self) -> None:
        """Log the current epoch."""
        self.log("epoch", self.trainer.current_epoch)

    def _shared_step(self, batch):
        """The shared step for the training and validation steps."""
        x = batch
        n = x.shape[0]
        t = torch.randint(1, self.diffusion_schedule.n_steps, (n,)).to(x.device)
        z, t, eps = self.add_noise(x, t)
        eps_tilde = self.denoiser(z, t)
        loss = F.mse_loss(eps_tilde, eps, reduction="none")
        return loss, t

    def add_noise(self, x, t):
        while len(t.shape) < len(x.shape):
            t = t.unsqueeze(-1)
        assert (t != 0).all()
        eps = torch.randn_like(x)
        schedule = self.diffusion_schedule(t)
        z = torch.sqrt(schedule["alpha"]) * x + torch.sqrt(schedule["1-alpha"]) * eps
        return z, t, eps

    def plot_snrs(self, loss_locs, loss_vals) -> plt.Figure:
        t, log_snr = self.log_signal_to_noise()
        t = t.detach().cpu().numpy()
        signal = torch.sqrt(torch.sigmoid(log_snr))
        noise = torch.sqrt(torch.sigmoid(-log_snr))

        fig, ax = plt.subplots(2, 2, figsize=(12, 16), sharex=True)
        ax[0, 0].plot(t, signal.detach().cpu().numpy(), label="signal")
        ax[0, 0].plot(t, noise.detach().cpu().numpy(), label="noise")
        ax[1, 0].plot(
            t, log_snr.detach().cpu().numpy(), label="log snr", color="k", lw=2
        )
        ax[1, 0].set_xlabel("t")
        ax[1, 0].set_ylabel("SNR")
        ax[0, 0].legend()

        log_snr_diffs = torch.diff(log_snr)
        incremental_noise = -torch.expm1(log_snr_diffs) * torch.sigmoid(-log_snr[1:])
        ax[0, 1].plot(t[1:], incremental_noise.detach().cpu().numpy(), label="beta")
        ax[0, 1].set_ylabel(r"$\beta$")
        ax[1, 1].scatter(
            x=loss_locs.detach().cpu().numpy().flatten(),
            y=loss_vals.detach().cpu().numpy().flatten(),
            alpha=0.5,
        )
        ax[1, 1].set_xlabel("t")
        ax[1, 1].set_ylabel("MSE Loss")
        # add a tick at the last time step:
        ax[1, 1].text(
            1,
            0.5,
            "noise",
            horizontalalignment="right",
            verticalalignment="center",
            transform=ax[1, 1].transAxes,
            fontsize=12,
            color="black",
        )
        ax[1, 1].text(
            0,
            0.5,
            "image",
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax[1, 1].transAxes,
            fontsize=12,
            color="black",
        )
        ax[1, 1].set_title("Losses by Diffusion Step")

        # Add a title to the figure:
        fig.suptitle("Signal, Noise, and Errors in the Diffusion Process")
        return fig

    def validation_step(self, batch, batch_idx):
        """The validation step for the diffusion model."""
        # Have the same seed as the batch, to keep
        # sample generation smooth across epochs:
        samples = self.generate(len(batch), seed=batch_idx)
        err = abs(samples.mean() - batch.mean())
        self.log("val/mean_err", err)
        e_coeff = energy_coefficient(samples, batch)
        self.log("val/energy_coeff", e_coeff)
        if batch_idx == 0 and self.sample_plotter is not None:
            fig = self.sample_plotter(batch, samples)
            self.log_image("val_images/samples", fig)

            self.log_histogram("val_images/samples_hist", samples.flatten())

            # Add a scatter plot of losses at each time step:

            loss, t = self._shared_step(batch)
            loss = loss.view(loss.size(0), -1).mean(1)

            fig = self.plot_snrs(t, loss)
            self.log_image("val_images/losses_by_time", fig)

            if self.noisy_image_plotter is not None:
                # Plot the noise:
                noisy_images = []
                for t in range(
                    1,
                    self.diffusion_schedule.n_steps,
                    self.diffusion_schedule.n_steps // 10,
                ):
                    z, *_ = self.add_noise(
                        batch[:10], t * torch.ones(10, dtype=torch.long)
                    )
                    noisy_images.append(z)
                fig = self.noisy_image_plotter(noisy_images)
                self.log_image(f"val_images/noise_{t}", fig)

        if self.sample_metrics is not None:
            import timeit

            # Time the metric update:
            for k, metric in self.sample_metrics.items():
                start = timeit.default_timer()
                metric.update(self.sample_metric_pre_process_fn(samples), real=False)
                metric.update(self.sample_metric_pre_process_fn(batch), real=True)
                end = timeit.default_timer()
                self.log(f"update_time/{k}", end - start)
        return err

    def log_image(self, name, fig):
        if hasattr(self.logger, "run"):
            self.logger.run[name].append(fig)
        else:
            self.logger.experiment.add_figure(
                name, fig, global_step=self.trainer.current_epoch
            )

    def log_histogram(self, name, values: torch.Tensor):
        try:
            # We're using tensorboard:
            self.logger.experiment.add_histogram(
                name, values, global_step=self.trainer.current_epoch
            )
        except:
            pass
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.hist(values.detach().cpu().numpy(), bins=100)
        self.log_image(f"{name}_plt", fig)

    def on_train_epoch_end(self) -> None:
        generative_var = self.generative_variance_at_zero_mean()
        self.log("train/generative_variance", generative_var)

    def on_validation_epoch_end(self):
        """Log the metrics at the end of the validation epoch."""
        if self.sample_metrics is not None:
            import timeit

            for k, metric in self.sample_metrics.items():

                start = timeit.default_timer()
                output = metric.compute()
                if isinstance(output, tuple):
                    self.log_dict({f"{k}_mean": output[0], f"{k}_std": output[1]})
                else:
                    self.log(k, output)
                metric.reset()
                end = timeit.default_timer()
                self.log(f"compute_time/{k}", end - start)

    def configure_optimizers(self):
        """The optimizer for the diffusion model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def generate(self, n, seed=None):
        """Generate samples from the diffusion model."""
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
        else:
            gen = None
        z = torch.randn(n, *self.hparams.latent_shape, generator=gen).to(self.device)
        for t in range(self.diffusion_schedule.n_steps - 1, 0, -1):
            t = torch.tensor(t).to(self.device)
            while len(t.shape) < len(z.shape):
                t = t.unsqueeze(-1)
            beta = self.diffusion_schedule(t)["beta"]
            mean = self.decoder(z, t)
            eps = torch.randn(*z.shape, generator=gen).to(self.device)
            z = mean + torch.sqrt(beta) * eps
        return mean
