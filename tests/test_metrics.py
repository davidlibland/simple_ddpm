import hypothesis
import hypothesis.extra.numpy as h_np
import hypothesis.strategies as h_strats
import torch

from simple_diffusion.metrics import _all_pairs_distance


@hypothesis.given(
    h_strats.tuples(
        h_strats.integers(min_value=1, max_value=10),
        h_strats.integers(min_value=1, max_value=10),
        h_np.array_shapes(min_dims=1, max_dims=3),
    ).flatmap(
        lambda x: h_strats.tuples(
            h_np.arrays(
                dtype=int,
                elements=h_strats.integers(min_value=-10, max_value=10),
                shape=(x[0], *x[2]),
            ),
            h_np.arrays(
                dtype=int,
                elements=h_strats.integers(min_value=-10, max_value=10),
                shape=(x[1], *x[2]),
            ),
        )
    )
)
def test_all_pairs_distance(x_y):
    """Tests that the pairwise distance computation is correct."""
    x, y = x_y
    x = torch.tensor(x)
    y = torch.tensor(y)
    result = _all_pairs_distance(x, y)
    assert result.shape == (x.shape[0], y.shape[0])
    # Compute the distances manually:
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            assert result[i, j] == torch.sum((x[i] - y[j]) ** 2)
