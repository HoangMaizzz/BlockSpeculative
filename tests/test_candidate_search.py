import torch

from block_spec.candidate_search import brute_force_topk, heap_topk_joint


def test_heap_equals_bruteforce_and_is_stable():
    logits = torch.tensor([[4.0, 2.0, 1.0], [0.5, 3.0, 2.0]])
    lp = torch.log_softmax(logits, -1)
    got = heap_topk_joint(lp, 3, 6)
    expected = brute_force_topk(lp, 3, 6)
    assert [(c.token_ids, c.drafter_log_score) for c in got.candidates] == expected
    assert len({c.token_ids for c in got.candidates}) == 6
    assert all(0 < x <= 1 for x in got.retained_mass_per_position)


def test_score_is_sum_of_token_scores():
    result = heap_topk_joint(torch.log_softmax(torch.randn(3, 5), -1), 4, 5)
    for c in result.candidates:
        assert abs(c.drafter_log_score - sum(c.drafter_token_logprobs)) < 1e-6

