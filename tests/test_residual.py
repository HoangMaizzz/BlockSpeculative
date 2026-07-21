import torch

from block_spec.distributions import compute_residual_distribution


def test_known_residual():
    r, fallback = compute_residual_distribution(torch.tensor([0.6, 0.4]), torch.tensor([0.2, 0.8]))
    assert not fallback
    assert torch.allclose(r, torch.tensor([1.0, 0.0]))
    assert (r >= 0).all() and torch.isclose(r.sum(), torch.tensor(1.0))


def test_equal_uses_p_fallback():
    p = torch.tensor([0.25, 0.75])
    r, fallback = compute_residual_distribution(p, p)
    assert fallback and torch.allclose(r, p)

