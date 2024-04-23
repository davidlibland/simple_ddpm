"""A simple diffusion model"""

import math
from typing import Dict, Callable, Tuple

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from simple_diffusion.fully_connected_denoiser import Denoiser as FC_Denoiser
from simple_diffusion.metrics import energy_coefficient
from simple_diffusion.unet import UNet


class AbsDiffusionSchedule(nn.Module):
    """An abstract diffusion schedule."""

    def log_snr(self, t):
        """Returns the log signal to noise ratio."""
        raise NotImplementedError

    def dlog_snr(self, t):
        """Returns the slope of the log signal to noise ratio."""
        raise NotImplementedError

    def normalized_log_snr(self, t):
        """Returns the normalized log signal to noise ratio."""
        log_snr = self.log_snr(t)
        return (log_snr - self.log_snr_min) / (self.log_snr_max - self.log_snr_min)

    def forward(self, t, s=None):
        """Returns the signal at time t"""
        gamma_t = self.log_snr(t)
        sigma_2_t = torch.sigmoid(gamma_t)
        alpha_2_t = torch.sigmoid(-gamma_t)
        alpha_t = torch.sqrt(alpha_2_t)
        sigma_t = torch.sqrt(sigma_2_t)
        result = {
            "alpha_t": alpha_t,
            "sigma_t": sigma_t,
        }
        if s is not None:
            gamma_s = self.log_snr(s)
            log_alpha_t = -F.softplus(gamma_t) / 2
            log_alpha_s = -F.softplus(gamma_s) / 2
            alpha_s_t = torch.exp(log_alpha_s - log_alpha_t)
            sigma_2_s = torch.sigmoid(gamma_s)
            sigma_s = torch.sqrt(sigma_2_s)
            expm1_delta = -torch.expm1(gamma_s - gamma_t)
            result["alpha_s_t"] = alpha_s_t
            result["sigma_s"] = sigma_s
            result["expm1_delta"] = expm1_delta
        return result


class LinearDiffusionSchedule(AbsDiffusionSchedule):
    def __init__(self):
        super().__init__()
        # beta_schedule = torch.linspace(beta_min, beta_max, n_steps, dtype=torch.float)
        # alpha_schedule = torch.cumprod(1 - beta_schedule, dim=0)  # (T,)
        # self.register_buffer("alpha_schedule", alpha_schedule)
        # self.register_buffer("beta_schedule", beta_schedule)
        # signal = torch.sqrt(self.alpha_schedule)
        # noise = 1 - self.alpha_schedule
        # log_snr = torch.log(torch.square(signal) / noise)
        # self.register_buffer("log_snr", log_snr)

    def log_snr(self, t):
        """Returns the log signal to noise ratio."""
        log_snr = torch.log(torch.expm1(1e-4 + 10 * t**2))
        return log_snr

    def normalized_log_snr(self, t):
        """Returns the log signal to noise ratio."""
        log_snr_min = math.log(math.expm1(1e-4))
        log_snr_max = math.log(math.expm1(1e-4 + 10))
        log_snr = self.log_snr(t)
        return (log_snr - log_snr_min) / (log_snr_max - log_snr_min)

    def dlog_snr(self, t):
        """Returns the derivative of the log signal to noise ratio."""
        dlog_snr = 2 * 10 * torch.exp(1e-4 + 10 * t**2) / torch.expm1(1e-4 + 10 * t**2)
        return dlog_snr


class LogitLinearSNR(AbsDiffusionSchedule):
    def __init__(self, log_snr_min=-6, log_snr_max=6):
        super().__init__()
        self.log_snr_min = log_snr_min
        self.log_snr_max = log_snr_max

    def log_snr(self, t):
        """Returns the log signal to noise ratio."""
        return t * (self.log_snr_max - self.log_snr_min) + self.log_snr_min

    def dlog_snr(self, t):
        """Returns the log signal to noise ratio."""
        return (self.log_snr_max - self.log_snr_min) * torch.ones_like(t)


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
        ema_decay=0.9999,
        n_time_steps=100,
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
        self.diffusion_schedule: AbsDiffusionSchedule = self._build_diffusion_schedule(
            **diffusion_schedule_kwargs
        )
        self.denoiser = self._build_denoiser(**denoiser_kwargs)
        self.sample_plotter = sample_plotter
        self.sample_metrics = sample_metrics
        self.sample_metric_pre_process_fn = sample_metric_pre_process_fn
        self.noisy_image_plotter = noisy_image_plotter
        self.ema = ExponentialMovingAverage(self.denoiser.parameters(), decay=ema_decay)

    def log_signal_to_noise(self):
        """Compute the log signal to noise ratio of the denoiser."""
        t = torch.linspace(0, 1, self.hparams.n_time_steps)
        return t, self.diffusion_schedule.log_snr(t)

    def generative_variance_at_zero_mean(self):
        """Compute the variance at time 0."""
        var = 1
        # Fixme
        # for t in torch.linspace(1, 0, self.hparams.n_time_steps).to(self.device):
        #     beta = self.diffusion_schedule(t)["beta"].detach().cpu().numpy()
        #     one_m_beta = self.diffusion_schedule(t)["1-beta"].detach().cpu().numpy()
        #     var = var / one_m_beta + beta
        return var

    def _build_denoiser(self, **denoiser_kwargs):
        """Build the denoiser network."""
        denoiser_kwargs = {**denoiser_kwargs}
        denoiser_type = denoiser_kwargs.pop("type", "fully_connected")
        if denoiser_type == "fully_connected":
            return FC_Denoiser(
                latent_shape=self.hparams.latent_shape,
                **denoiser_kwargs,
            )
        elif denoiser_type == "unet":
            return UNet(
                **denoiser_kwargs,
            )

    def _build_diffusion_schedule(
        self, schedule_type="linear", **kwargs
    ) -> AbsDiffusionSchedule:
        """Build the denoiser network."""
        if schedule_type == "linear":
            return LinearDiffusionSchedule(**kwargs)
        elif schedule_type == "logit_linear":
            return LogitLinearSNR(**kwargs)
        else:
            raise NotImplementedError(f"Schedule type {schedule_type} not implemented.")

    def decoder(self, z, t, s):
        """The decoder network, defined in terms of the denoiser."""
        schedule = self.diffusion_schedule(t, s)
        sigma_t = schedule["sigma_t"]
        alpha_s_t = schedule["alpha_s_t"]
        expm1_delta = schedule["expm1_delta"]
        noise_est = self.denoiser(z, t.expand(z.shape[0], *t.shape[1:]))

        mean = alpha_s_t * (z - sigma_t * expm1_delta * noise_est)
        return mean
        # factor = beta / torch.sqrt(schedule["1-alpha"])
        # return (z - factor * noise_est) / torch.sqrt(schedule["1-beta"])

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
        n = x.size(0)
        u = torch.rand(1).to(x.device)
        i = torch.arange(n, device=x.device) / n
        t = (i + u) % 1
        z, t, eps = self.add_noise(x, t)
        gamma_t_norm = self.diffusion_schedule.normalized_log_snr(t)
        gamma_t_prime = self.diffusion_schedule.dlog_snr(t)
        eps_tilde = self.denoiser(z, gamma_t_norm)
        loss = F.mse_loss(eps_tilde, eps, reduction="none")
        # Rescale the loss by the derivative of the log snr:
        while len(gamma_t_prime.shape) < len(loss.shape):
            gamma_t_prime = gamma_t_prime.unsqueeze(-1)
        loss = gamma_t_prime * loss
        return loss, t

    def add_noise(self, x, t):
        while len(t.shape) < len(x.shape):
            t = t.unsqueeze(-1)
        assert (t != 0).all()
        eps = torch.randn_like(x)
        schedule = self.diffusion_schedule(t)
        z = schedule["alpha_t"] * x + schedule["sigma_t"] * eps
        return z, t, eps

    def plot_snrs(self, loss_locs, loss_vals) -> plt.Figure:
        t, neg_log_snr = self.log_signal_to_noise()
        log_snr = -neg_log_snr
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

        loss, t = self._shared_step(batch)
        self.log("val/loss", loss.mean(), prog_bar=True)
        generate_samples = any(
            [
                self.sample_metrics is not None,
                batch_idx == 0 and self.sample_plotter is not None,
            ]
        )
        if generate_samples:
            # Have the same seed as the batch, to keep
            # sample generation smooth across epochs:
            samples = self.generate(len(batch), seed=batch_idx)
            err = abs(samples.mean() - batch.mean())
            self.log("val/mean_err", err)
            e_coeff = energy_coefficient(samples, batch)
            self.log("val/energy_coeff", e_coeff)

            if self.sample_metrics is not None:
                import timeit

                # Time the metric update:
                for k, metric in self.sample_metrics.items():
                    start = timeit.default_timer()
                    metric.update(
                        self.sample_metric_pre_process_fn(samples), real=False
                    )
                    metric.update(self.sample_metric_pre_process_fn(batch), real=True)
                    end = timeit.default_timer()
                    self.log(f"update_time/{k}", end - start)

            if batch_idx == 0 and self.sample_plotter is not None:
                fig = self.sample_plotter(batch, samples)
                self.log_image("val_images/samples", fig)

                self.log_histogram("val_images/samples_hist", samples.flatten())

                # Add a scatter plot of losses at each time step:
                loss = loss.view(loss.size(0), -1).mean(1)

                fig = self.plot_snrs(t, loss)
                self.log_image("val_images/losses_by_time", fig)

                if self.noisy_image_plotter is not None:
                    # Plot the noise:
                    noisy_images = []
                    for t in torch.linspace(0, 1, 10, device=self.device):
                        if t == 0:
                            z = batch[:10]
                        else:
                            z, *_ = self.add_noise(batch[:10], t.expand(10))
                        noisy_images.append(z)
                    fig = self.noisy_image_plotter(noisy_images)
                    self.log_image(f"val_images/noise_{t}", fig)
        return loss.mean()

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

    def on_train_start(self) -> None:
        self.ema.to(self.device)

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.denoiser.parameters())

    def generate(self, n, seed=None):
        """Generate samples from the diffusion model."""
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
        else:
            gen = None
        eval_mode = self.training
        self.eval()
        self.ema.to(self.device)
        with self.ema.average_parameters():
            z = torch.randn(n, *self.hparams.latent_shape, generator=gen).to(
                self.device
            )
            ts = torch.linspace(1, 0, self.hparams.n_time_steps, device=self.device)
            for t, s in zip(ts, ts[1:]):
                while len(t.shape) < len(z.shape):
                    t = t.unsqueeze(-1)
                    s = s.unsqueeze(-1)
                schedule = self.diffusion_schedule(t, s)
                sigma_s = schedule["sigma_s"]
                expm1_delta = schedule["expm1_delta"]
                mean = self.decoder(z, t, s)
                eps = torch.randn(*z.shape, generator=gen).to(self.device)
                z = mean + sigma_s * torch.sqrt(expm1_delta) * eps
        self.train(eval_mode)
        return mean
