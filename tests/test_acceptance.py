import torch

from block_spec.acceptance import decide_acceptance, select_proposal


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

