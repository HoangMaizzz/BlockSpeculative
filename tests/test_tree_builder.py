import torch

from block_spec.drafter_adapter import DraftCandidates
from block_spec.tree_builder import AsymmetricTreeBuilder
from block_spec.types import BlockCandidate


class DummyDrafter:
    block_size = 2
    def propose(self, prefix):
        candidates = [BlockCandidate((i, i + 10), (-0.1 - i, -0.2), -0.3 - i) for i in range(5)]
        return DraftCandidates(candidates, (1.0, 1.0), 0.9)


def test_widths_quota_caps_and_budgets():
    builder = AsymmetricTreeBuilder(DummyDrafter(), [5, 20, 10, 5, 2, 1], [5, 5, 3, 2, 1, 1],
                                    2, max_tree_nodes=64, max_tree_tokens=126)
    tree = builder.build(torch.tensor([99]))
    assert all(a <= b for a, b in zip(tree.widths(), [5, 20, 10, 5, 2, 1]))
    assert len(tree.nodes) <= 64
    assert (len(tree.nodes) - 1) * 2 <= 126
    depth1 = [n for n in tree.nodes.values() if n.depth == 1]
    assert all(len(n.children) >= 2 for n in depth1)
    assert all(len(n.expanded_candidate_indices) <= 5 for n in depth1)
    assert all(len(n.candidate_set) == 5 for n in depth1)

