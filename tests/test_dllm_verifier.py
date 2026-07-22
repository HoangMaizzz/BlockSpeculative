import torch

from block_spec.dllm_verifier import FastDLLMVerifier
from block_spec.types import BlockCandidate, DraftTreeNode


class FakeAdapter:
    block_size = 2

    def __init__(self):
        self.calls = 0

    def marginal_log_probs(self, prefix):
        self.calls += 1
        return torch.log_softmax(
            torch.tensor([[4.0, 3.0, 0.0], [1.0, 5.0, 2.0]]), dim=-1
        )

    def marginal_log_probs_batch(self, prefixes):
        return torch.stack([self.marginal_log_probs(prefix) for prefix in prefixes])


class TinyTree:
    root_id = 0

    def __init__(self):
        self.nodes = {
            0: DraftTreeNode(
                node_id=0,
                parent_id=None,
                depth=0,
                block_token_ids=None,
                local_drafter_log_score=0.0,
                cumulative_drafter_log_score=0.0,
                confidence=1.0,
                candidate_set=[BlockCandidate((0, 1), (-0.1, -0.2), -0.3)],
            )
        }


def test_dllm_union_search_and_score_reuse_one_masked_forward():
    adapter = FakeAdapter()
    verifier = FastDLLMVerifier(
        adapter, topk=2, per_position_topk=2, node_batch_size=2
    )
    tree = TinyTree()
    beams, search = verifier.search_tree(torch.tensor([2]), tree)
    assert search["verifier_forward_calls"] == 1
    assert len(beams[0]) == 2
    report = verifier.score_tree(torch.tensor([2]), tree)
    assert adapter.calls == 1
    assert report["verifier_forward_calls"] == 0
    assert tree.nodes[0].candidate_set[0].verifier_log_score is not None
    assert report["coverage_is_full_vocab_exact"] is True
