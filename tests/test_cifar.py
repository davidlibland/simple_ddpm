from simple_diffusion.cifar.train import train


def test_cifar():
    """A smoke test for the cifar diffusion model."""
    train(
        n_steps=10,
        n_epochs=1,
        batch_size=100,
        u_steps=1,
        step_depth=1,
        initial_hidden=1,
        check_val_every_n_epoch=1,
        debug=True,
        beta_schedule_form="linear",
        n_groups=1,
        n_heads=1,
        cache=False,
    )
