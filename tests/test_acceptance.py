import torch

from block_spec.acceptance import (
    decide_acceptance,
    decide_self_selection_acceptance,
    select_proposal,
)
from block_spec.distributions import compute_self_selection_replacement


def test_monte_carlo_accept_residual_matches_p():
    p, q = torch.tensor([0.8, 0.2]), torch.tensor([0.3, 0.7])
    gen = torch.Generator().manual_seed(123)
    counts = torch.zeros(2)
    residual = torch.tensor([1.0, 0.0])
    for _ in range(20000):
        i = select_proposal(q, "sample_q", gen)
        d = decide_acceptance(i, (i,), p, q, gen)
        j = i if d.accepted else int(torch.multinomial(residual, 1, generator=gen))
        counts[j] += 1
    empirical = counts / counts.sum()
    assert torch.allclose(empirical, p, atol=0.015)


def test_monte_carlo_self_selection_matches_p():
    p = torch.tensor([0.55, 0.30, 0.15])
    q = torch.tensor([0.10, 0.20, 0.70])
    gen = torch.Generator().manual_seed(321)
    counts = torch.zeros(3)
    for _ in range(30000):
        proposed = select_proposal(q, "sample_q", gen)
        decision = decide_self_selection_acceptance(
            proposed, (proposed,), p, q, gen
        )
        if decision.accepted:
            committed = proposed
        else:
            replacement, fallback = compute_self_selection_replacement(p, proposed)
            assert not fallback
            assert replacement[proposed] == 0
            committed = int(torch.multinomial(replacement, 1, generator=gen))
        counts[committed] += 1
    empirical = counts / counts.sum()
    assert torch.allclose(empirical, p, atol=0.015)
