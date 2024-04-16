import pytest
import torch.testing

from simple_diffusion.model import LinearDiffusionSchedule


@pytest.mark.parametrize(
    "n_steps, beta_min, beta_max", [(10, 1e-4, 0.02), (100, 1e-4, 0.02)]
)
def test_linear_diffusion_schedule(n_steps, beta_min, beta_max):
    schedule = LinearDiffusionSchedule(
        n_steps=n_steps, beta_min=beta_min, beta_max=beta_max
    )
    for t in range(1, n_steps):
        result = schedule(t)
        alpha = result["alpha"]
        beta = result["beta"]
        alpha_expected = schedule.alpha_schedule[t]
        beta_expected = schedule.beta_schedule[t]
        torch.testing.assert_allclose(
            alpha, alpha_expected, msg=f"alpha={alpha}, expected={alpha_expected}"
        )
        torch.testing.assert_allclose(
            beta, beta_expected, msg=f"beta={beta}, expected={beta_expected}"
        )
