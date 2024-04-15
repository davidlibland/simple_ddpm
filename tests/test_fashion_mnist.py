from simple_diffusion.fashion_mnist.train import train


def test_fashion_mnist():
    """A smoke test for the fashion mnist diffusion model."""
    train(
        n_steps=100,
        n_epochs=1,
        log_to_neptune=False,
        batch_size=100,
        check_val_every_n_epoch=1,
        debug=True,
        beta_schedule_form="linear",
    )
