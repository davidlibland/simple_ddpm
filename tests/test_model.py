import pytest
import torch.testing
import torch.nn.functional as F


@pytest.mark.parametrize("x_", range(-50, 50))
def test_log_softplus(x_):
    x = torch.tensor(x_, dtype=torch.float32) / 10
    expected = torch.log(torch.sigmoid(x))
    actual = -F.softplus(-x)
    assert torch.allclose(expected, actual, atol=1e-5)
