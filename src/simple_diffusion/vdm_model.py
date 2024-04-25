"""A simple diffusion model"""

from typing import Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from torchmetrics import Metric

from simple_diffusion.fully_connected_denoiser import Denoiser as FC_Denoiser
from simple_diffusion.model_base import BaseDiffusionModel
from simple_diffusion.unet import UNet


class DiffusionModel(BaseDiffusionModel):
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
        super().__init__(
            sample_plotter=sample_plotter,
            learning_rate=learning_rate,
            sample_metrics=sample_metrics,
            sample_metric_pre_process_fn=sample_metric_pre_process_fn,
            diffusion_schedule_kwargs=diffusion_schedule_kwargs,
            noisy_image_plotter=noisy_image_plotter,
            ema_decay=ema_decay,
            n_time_steps=n_time_steps,
        )
        self.save_hyperparameters(
            ignore=[
                "sample_plotter",
                "sample_metrics",
                "sample_metric_pre_process_fn",
                "noisy_image_plotter",
                "learning_rate",
                "diffusion_schedule_kwargs",
                "ema_decay",
                "n_time_steps",
            ]
        )
        self.denoiser = self._build_denoiser(**denoiser_kwargs)
        self.configure_model_base()

    def trainable_parameters(self):
        """The trainable parameters of the model."""
        return self.denoiser.parameters()

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

    def decoder(self, z, t, s):
        """The decoder network, defined in terms of the denoiser."""
        schedule = self.diffusion_schedule(t, s)
        sigma_t = schedule["sigma_t"]
        alpha_s_t = schedule["alpha_s_t"]
        expm1_delta = schedule["expm1_delta"]
        noise_est = self.denoiser(z, t.expand(z.shape[0], *t.shape[1:]))

        mean = alpha_s_t * (z - sigma_t * expm1_delta * noise_est)
        return mean

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
        loss_dict = {
            "total_loss": loss.mean(),
            "diffusion_loss": loss.view(n, -1).mean(1),
            "t": t,
        }
        return loss_dict

    def add_noise(self, x, t):
        while len(t.shape) < len(x.shape):
            t = t.unsqueeze(-1)
        eps = torch.randn_like(x)
        schedule = self.diffusion_schedule(t)
        z = schedule["alpha_t"] * x + schedule["sigma_t"] * eps
        return z, t, eps

    def _generate_samples(self, n, gen):
        """Generate samples from the diffusion model."""
        z = torch.randn(n, *self.hparams.latent_shape, generator=gen).to(self.device)
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
        return mean
