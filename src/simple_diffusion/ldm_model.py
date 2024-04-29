"""A simple diffusion model"""

from typing import Dict, Callable

import torch
import torch.nn.functional as F
from torchmetrics import Metric

from simple_diffusion.conv_vae import ConvEncoder, ConvDecoder
from simple_diffusion.fully_connected_denoiser import Denoiser as FC_Denoiser
from simple_diffusion.fully_connected_vae import (
    Encoder as FCEncoder,
    Decoder as FCDecoder,
)
from simple_diffusion.model_base import BaseDiffusionModel
from simple_diffusion.unet import UNet


class LatentDiffusionModel(BaseDiffusionModel):
    def __init__(
        self,
        sample_plotter=None,
        learning_rate=1e-1,
        sample_metrics: Dict[str, Metric] = None,
        sample_metric_pre_process_fn: Callable = None,
        diffusion_schedule_kwargs: Dict = {},
        encoder_kwargs: Dict = {},
        decoder_kwargs: Dict = {},
        noisy_image_plotter=None,
        ema_decay=0.9999,
        n_time_steps=100,
        denoiser_kwargs: Dict = {},
        vae_weight: float = 1.0,
        latent_dim=8,
    ):
        """
        A simple diffusion model.
        Args:
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
        self.vae_encoder = self._build_encoder(**encoder_kwargs)
        self.vae_decoder = self._build_decoder(**decoder_kwargs)
        self.configure_model_base()

    def trainable_parameters(self):
        """The trainable parameters of the model."""
        return (
            list(self.denoiser.parameters())
            + list(self.vae_encoder.parameters())
            + list(self.vae_decoder.parameters())
        )

    def _build_denoiser(self, **denoiser_kwargs):
        """Build the denoiser network."""
        denoiser_kwargs = {**denoiser_kwargs}
        denoiser_type = denoiser_kwargs.pop("type", "fully_connected")
        if denoiser_type == "fully_connected":
            return FC_Denoiser(
                latent_shape=(self.hparams.latent_dim,), **denoiser_kwargs
            )
        elif denoiser_type == "unet":
            return UNet(
                **denoiser_kwargs,
            )

    def _build_encoder(self, **encoder_kwargs):
        """Build the denoiser network."""
        encoder_kwargs = {**encoder_kwargs}
        encoder_type = encoder_kwargs.pop("type", "fully_connected")
        if encoder_type == "fully_connected":
            return FCEncoder(
                latent_dim=self.hparams.latent_dim,
                **encoder_kwargs,
            )
        if encoder_type == "conv":
            return ConvEncoder(
                latent_dim=self.hparams.latent_dim,
                **encoder_kwargs,
            )

    def _build_decoder(self, **decoder_kwargs):
        """Build the denoiser network."""
        decoder_kwargs = {**decoder_kwargs}
        decoder_type = decoder_kwargs.pop("type", "fully_connected")
        if decoder_type == "fully_connected":
            return FCDecoder(
                latent_dim=self.hparams.latent_dim,
                **decoder_kwargs,
            )
        if decoder_type == "conv":
            return ConvDecoder(
                latent_dim=self.hparams.latent_dim,
                **decoder_kwargs,
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

    def diffusion_loss(self, x_latent) -> Dict[str, torch.Tensor]:
        """The diffusion loss."""
        n = x_latent.size(0)
        u = torch.rand(1, 1).to(x_latent.device)
        i = (torch.arange(n, device=x_latent.device) / n).view(-1, 1)
        t = (i + u) % 1
        z, t, eps = self.add_noise(x_latent, t)
        gamma_t_norm = self.diffusion_schedule.normalized_log_snr(t)
        gamma_t_prime = self.diffusion_schedule.dlog_snr(t)
        eps_tilde = self.denoiser(z, gamma_t_norm)
        loss = F.mse_loss(eps_tilde, eps, reduction="none")
        # Rescale the loss by the derivative of the log snr:
        loss = gamma_t_prime * loss
        # Note that the total integral has bounds given by the lower and upper snr
        # To control for that we rescale the loss appropriately.
        loss = loss / (
            self.diffusion_schedule.log_snr_max - self.diffusion_schedule.log_snr_min
        )
        return {"loss": loss.sum(dim=1), "t": t}

    def reconstruction_loss(self, x, x_latent) -> torch.Tensor:
        """
        The reconstruction loss term from the VAE

        Args:
            x (torch.Tensor): The input data. Shape (batch, *input_shape)
            x_latent (torch.Tensor): The mean of the latent space. Shape (batch, latent)
        """
        x_recon = self.vae_decoder(x_latent)
        return -x_recon.log_prob(x)

    def latent_loss(self, x_latent_mean) -> Dict[str, torch.Tensor]:
        """
        The KL loss term from the VAE

        Args:
            x_latent_mean (torch.Tensor): The mean of the latent space. Shape (batch, latent)
        """
        n = x_latent_mean.size(0)
        t = torch.ones(n, 1, device=x_latent_mean.device)
        s = torch.zeros(n, 1, device=x_latent_mean.device)
        schedule = self.diffusion_schedule(t, s)
        alpha_t = schedule["alpha_t"]
        sigma_t = schedule["sigma_t"]
        sigma_s = schedule["sigma_s"]
        mean = alpha_t * x_latent_mean
        var = sigma_t**2 + alpha_t**2 * sigma_s**2
        kl_divergence = -torch.log(var) / 2 + (mean**2 + var) / 2 - 0.5
        standard_scale_factor = 1 / alpha_t**2
        kls = kl_divergence.sum(dim=1)
        return {
            "latent_loss": kls,
            "standardized_kls": kls * standard_scale_factor,
        }

    def _shared_step(self, batch):
        """The shared step for the training and validation steps."""
        x = batch
        x_latent_mean = self.vae_encoder(x)
        self.log("train/latent_std", x_latent_mean.std())

        n = x_latent_mean.size(0)
        t = torch.zeros(n, 1, device=x_latent_mean.device)
        x_latent, _, eps_0 = self.add_noise(x_latent_mean, t)

        # Compute the latent loss:
        latent_loss_dict = self.latent_loss(x_latent_mean)
        latent_loss = latent_loss_dict["latent_loss"]
        standardized_kls = latent_loss_dict["standardized_kls"]

        # Compute the reconstruction loss:
        reconstruction_loss = self.reconstruction_loss(x, x_latent)

        # Compute the diffusion loss:
        diffusion_loss_dict = self.diffusion_loss(x_latent_mean)

        # To get the actual elbo we need to scale the
        # diffusion loss by the number of time steps
        # This is because we are taking a single MC sample of the diffusion loss per
        # batch sample, but it appears n_time_steps times in the loss.
        diffusion_scale_factor = self.hparams.n_time_steps
        # Combine the losses:
        loss = (
            standardized_kls.mean() * self.hparams.vae_weight
            + reconstruction_loss.mean() * self.hparams.vae_weight
            + diffusion_loss_dict["loss"].mean() * diffusion_scale_factor
        )
        elbo = (
            latent_loss.mean()
            + reconstruction_loss.mean()
            + diffusion_loss_dict["loss"].mean() * diffusion_scale_factor
        ).detach()

        self.log("train/latent_loss", latent_loss.mean(), prog_bar=True)
        self.log("train/reconstruction_loss", reconstruction_loss.mean(), prog_bar=True)
        self.log(
            "train/diffusion_loss",
            diffusion_loss_dict["loss"].mean(),
            prog_bar=True,
        )
        loss_dict = {
            "total_loss": loss,
            "diffusion_loss": diffusion_loss_dict["loss"],
            "t": diffusion_loss_dict["t"],
            "reconstruction_loss": reconstruction_loss.mean(),
            "latent_loss": latent_loss.mean(),
            "elbo": elbo,
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
        z = torch.randn(n, self.hparams.latent_dim, generator=gen).to(self.device)
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
        samples = self.vae_decoder(z).sample()
        return samples
