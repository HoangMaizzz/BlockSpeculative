import torch

from block_spec.distributions import normalize_log_scores


def test_stable_very_negative_scores():
    p = normalize_log_scores(torch.tensor([-10000.0, -10001.0, -10002.0]))
    assert torch.isfinite(p).all()
    assert abs(p.sum().item() - 1) < 1e-6

