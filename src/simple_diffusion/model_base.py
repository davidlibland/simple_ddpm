"""A simple diffusion model"""

from typing import Dict, Callable, Tuple

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from simple_diffusion.metrics import energy_coefficient
from simple_diffusion.schedules import (
    AbsDiffusionSchedule,
    LinearDiffusionSchedule,
    LogitLinearSNR,
)


class BaseDiffusionModel(L.LightningModule):
    def __init__(
        self,
        sample_plotter=None,
        learning_rate=1e-1,
        sample_metrics: Dict[str, Metric] = None,
        sample_metric_pre_process_fn: Callable = None,
        diffusion_schedule_kwargs: Dict = {},
        noisy_image_plotter=None,
        ema_decay=0.9999,
        n_time_steps=100,
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
                "sample_metrics",
                "sample_metric_pre_process_fn",
                "noisy_image_plotter",
            ]
        )
        self.diffusion_schedule: AbsDiffusionSchedule = self._build_diffusion_schedule(
            **diffusion_schedule_kwargs
        )
        self.sample_plotter = sample_plotter
        self.sample_metrics = (
            None if sample_metrics is None else nn.ModuleDict(sample_metrics)
        )
        self.sample_metric_pre_process_fn = sample_metric_pre_process_fn
        self.noisy_image_plotter = noisy_image_plotter

    def configure_model_base(self):
        """Configure the model base."""
        self.ema = ExponentialMovingAverage(
            self.trainable_parameters(), decay=self.hparams.ema_decay
        )

    def trainable_parameters(self):
        """The trainable parameters of the model."""
        raise NotImplementedError

    def log_signal_to_noise(self):
        """Compute the log signal to noise ratio of the denoiser."""
        t = torch.linspace(0, 1, self.hparams.n_time_steps)
        return t, self.diffusion_schedule.log_snr(t)

    def generative_variance_at_zero_mean(self):
        """Compute the variance at time 0."""
        var = 1
        ts = torch.linspace(1, 0, self.hparams.n_time_steps, device=self.device)
        for t, s in zip(ts, ts[1:]):
            schedule = self.diffusion_schedule(t, s)
            sigma_s = schedule["sigma_s"]
            expm1_delta = schedule["expm1_delta"]
            alpha_s_t = schedule["alpha_s_t"]

            var = alpha_s_t**2 * var + sigma_s**2 * expm1_delta
        return var

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

    def configure_optimizers(self):
        """The optimizer for the diffusion model."""
        optimizer = torch.optim.Adam(
            self.trainable_parameters(), lr=self.hparams.learning_rate
        )
        return optimizer

    def training_step(self, batch):
        """The training step for the diffusion model."""
        loss_dict = self._shared_step(batch)
        loss = loss_dict["total_loss"]
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _shared_step(self, batch) -> Dict[str, torch.Tensor]:
        """The shared step for the training and validation steps."""
        raise NotImplementedError

    def on_train_epoch_start(self) -> None:
        """Log the current epoch."""
        self.log("epoch", self.trainer.current_epoch)

    def _shared_step(self, batch):
        """The shared step for the training and validation steps."""
        raise NotImplementedError

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

        loss_dict = self._shared_step(batch)
        total_loss = loss_dict["total_loss"]
        diffusion_loss = loss_dict["diffusion_loss"]
        t = loss_dict["t"]
        self.log("val/loss", total_loss, prog_bar=True)
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

                fig = self.plot_snrs(t, diffusion_loss)
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
        return total_loss

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

    def on_train_start(self) -> None:
        self.ema.to(self.device)

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.trainable_parameters())

    def _generate_samples(self, n, gen: torch.Generator):
        """Generate samples from the diffusion model."""
        raise NotImplementedError

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
            samples = self._generate_samples(n, gen)
        self.train(eval_mode)
        return samples
