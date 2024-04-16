from simple_diffusion.cifar.train import train


def test_cifar():
    """A smoke test for the cifar diffusion model."""
    train(
        n_steps=100,
        n_epochs=1,
        batch_size=100,
        check_val_every_n_epoch=1,
        debug=True,
        beta=0.02,
        beta_schedule_form="linear",
    )
