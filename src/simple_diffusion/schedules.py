import math

import torch
from torch import nn as nn
from torch.nn import functional as F


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
