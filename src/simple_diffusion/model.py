"""A simple diffusion model"""

import lightning as L
import torch
import torch.nn.functional as F

from simple_diffusion.fully_connected_denoiser import Denoiser as FC_Denoiser
from simple_diffusion.metrics import energy_coefficient


class DiffusionModel(L.LightningModule):
    def __init__(self, beta_schedule, sample_plotter=None, **denoiser_kwargs):
        """
        A simple diffusion model.
        Args:
            beta_schedule (torch.Tensor): The schedule of beta values.
            sample_plotter (callable): A function that plots samples, returns a figure.
            denoiser_kwargs (dict): The keyword arguments for the denoiser.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["sample_plotter"])
        self.register_buffer("beta_schedule", beta_schedule)  # (T,)
        # assert self.beta_schedule[0] == 0
        alpha_schedule = torch.cumprod(1 - beta_schedule, dim=0)  # (T,)
        self.register_buffer("alpha_schedule", alpha_schedule)
        self.denoiser = self._build_denoiser(**denoiser_kwargs)
        self.latent_shape = denoiser_kwargs["latent_shape"]
        self.sample_plotter = sample_plotter

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
            raise NotImplementedError("UNet denoiser not implemented.")

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
        loss = F.mse_loss(eps_tilde, eps)
        self.logger.log_metrics({"train/loss": loss})
        return loss

    def validation_step(self, batch):
        """The validation step for the diffusion model."""
        samples = self.generate(len(batch))
        err = abs(samples.mean() - batch.mean())
        self.logger.log_metrics({"val/mean_err": err})
        e_coeff = energy_coefficient(samples, batch)
        self.logger.log_metrics({"val/energy_coeff": e_coeff})
        if self.sample_plotter is not None:
            fig = self.sample_plotter(batch, samples)
            self.logger.run["val/samples"].append(fig)
        return err

    def configure_optimizers(self):
        """The optimizer for the diffusion model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
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
