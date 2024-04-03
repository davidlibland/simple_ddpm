from simple_diffusion.gaussian_1d.train import train


def test_1d_gaussian():
    """A smoke test for the 1D Gaussian diffusion model."""
    train(
        n_steps=100,
        n_epochs=1,
        log_to_neptune=False,
        n_samples=100,
        batch_size=100,
    )
