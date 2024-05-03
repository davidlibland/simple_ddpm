import torch


def test_detach():
    x = torch.tensor([1.0], requires_grad=True)

    y = x**2
    z = -((x.detach()) ** 2)
    w = y + z
    w.backward()
    assert x.grad.item() == 2.0
