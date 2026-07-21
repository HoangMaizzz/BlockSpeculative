from types import SimpleNamespace

import torch

from block_spec.candidate_search import score_block_from_marginals
from block_spec.distributions import compute_residual_distribution
from block_spec.tree import DraftTree
from block_spec.tree_ar_scorer import ARTreeScorer
from block_spec.types import BlockCandidate, DraftTreeNode
from block_spec.verifier_candidate_search import (
    ARVerifierTopKBlockSearch,
    merge_verifier_topk_into_tree,
)


class TinyBase(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        self.forward_calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyVerifier(torch.nn.Module):
    def __init__(self, vocab_size=7, hidden_size=5):
        super().__init__()
        torch.manual_seed(4)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.model = TinyBase(self.embedding)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embedding


def test_saved_marginals_score_arbitrary_block():
    table = torch.log_softmax(torch.arange(14, dtype=torch.float32).reshape(2, 7), dim=-1).half()
    token_scores, block_score = score_block_from_marginals(table, (1, 6))
    assert len(token_scores) == 2
    assert abs(block_score - sum(token_scores)) < 1e-6
    assert abs(token_scores[0] - float(table[0, 1])) < 1e-6
    assert abs(token_scores[1] - float(table[1, 6])) < 1e-6


def test_verifier_beam_union_adds_blocks_and_looks_up_q():
    q_table = torch.log_softmax(torch.randn(2, 7), dim=-1).half()
    original = BlockCandidate((0, 0), (float(q_table[0, 0]), float(q_table[1, 0])), -3.0)
    root = DraftTreeNode(
        0, None, 0, None, 0.0, 0.0, 0.0,
        candidate_set=[original],
        drafter_marginal_logprobs=q_table,
    )
    tree = DraftTree({0: root})
    model = TinyVerifier()
    beams, diagnostics = ARVerifierTopKBlockSearch(
        model, block_size=2, topk=2, batch_size=4
    ).search_tree(torch.tensor([1, 2]), tree)
    assert len(beams[0]) == 2
    assert model.model.forward_calls == 2
    assert diagnostics["verifier_forward_calls"] == 2
    merged = merge_verifier_topk_into_tree(tree, beams)
    assert merged["verifier_only_blocks_added"] + merged["duplicate_blocks_merged"] == 2
    for candidate in root.candidate_set:
        expected_tokens, expected_joint = score_block_from_marginals(q_table, candidate.token_ids)
        if candidate.candidate_source == "verifier":
            assert candidate.drafter_token_logprobs == expected_tokens
            assert abs(candidate.drafter_log_score - expected_joint) < 1e-6
    score_result = ARTreeScorer(model).score_tree(torch.tensor([1, 2]), tree)
    assert score_result["restricted_lm_head"] is True
    assert abs(sum(candidate.q_normalized for candidate in root.candidate_set) - 1.0) < 1e-6
    assert abs(sum(candidate.p_normalized for candidate in root.candidate_set) - 1.0) < 1e-6
    q = torch.tensor([candidate.q_normalized for candidate in root.candidate_set])
    p = torch.tensor([candidate.p_normalized for candidate in root.candidate_set])
    residual, _ = compute_residual_distribution(p, q)
    assert residual.shape[0] == len(root.candidate_set)
    assert abs(float(residual.sum()) - 1.0) < 1e-6
