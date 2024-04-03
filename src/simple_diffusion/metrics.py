import torch


def energy_coefficient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the energy coeficient, cf https://en.wikipedia.org/wiki/Energy_distance"""
    n = min(x.shape[0], 100)
    m = min(y.shape[0], 100)
    A = _all_pairs_distance(x[:n], y[:m]).mean()
    B = _all_pairs_distance(x[:n], x[:n]).mean()
    C = _all_pairs_distance(y[:n], y[:m]).mean()
    e_coeff = (2 * A - B - C) / (2 * A)
    return e_coeff


def _all_pairs_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the pairwise distance between two sets of points."""
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    x_norm = (x**2).sum(dim=-1, keepdim=True)
    y_norm = (y**2).sum(dim=-1, keepdim=True)
    xy = x @ y.transpose(-2, -1)
    return x_norm + y_norm.transpose(-2, -1) - 2 * xy
