from types import SimpleNamespace

import torch

from block_spec.drafter_adapter import DraftCandidates
from block_spec.tree_ar_scorer import ARTreeScorer, build_tree_attention_layout
from block_spec.tree_builder import AsymmetricTreeBuilder
from block_spec.types import BlockCandidate


class TwoCandidateDrafter:
    block_size = 2

    def propose(self, prefix):
        return DraftCandidates(
            [
                BlockCandidate((3, 4), (-0.1, -0.2), -0.3),
                BlockCandidate((5, 6), (-1.0, -1.0), -2.0),
            ],
            (0.9, 0.9),
            0.75,
        )


class TinyTreeModel(torch.nn.Module):
    def __init__(self, vocab=12):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 6)
        self.head = torch.nn.Linear(6, vocab, bias=False)
        self.forward_calls = 0

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        self.forward_calls += 1
        assert attention_mask is not None and attention_mask.ndim == 4
        assert position_ids is not None
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def test_tree_mask_allows_ancestors_but_blocks_siblings():
    drafter = TwoCandidateDrafter()
    tree = AsymmetricTreeBuilder(
        drafter, [2, 1], [2, 1], 0, max_tree_nodes=4, max_tree_tokens=6
    ).build(torch.tensor([1, 2]))
    layout = build_tree_attention_layout(torch.tensor([1, 2]), tree)
    root_edges = [edge for edge in layout.edges if edge.parent_node_id == 0]
    child_node = tree.nodes[tree.root.children[0]]
    child_edge = next(edge for edge in layout.edges if edge.parent_node_id == child_node.node_id)
    incoming_edge = next(
        edge
        for edge in root_edges
        if tree.child_for_candidate(tree.root, edge.candidate_index) is child_node
    )
    sibling_edge = next(edge for edge in root_edges if edge is not incoming_edge)
    query = child_edge.start
    assert layout.allowed_attention[query, :2].all()
    assert layout.allowed_attention[query, incoming_edge.start:incoming_edge.end].all()
    assert not layout.allowed_attention[query, sibling_edge.start:sibling_edge.end].any()


def test_tree_scorer_uses_one_forward_and_populates_every_candidate():
    drafter = TwoCandidateDrafter()
    tree = AsymmetricTreeBuilder(
        drafter, [2, 1], [2, 1], 0, max_tree_nodes=4, max_tree_tokens=6
    ).build(torch.tensor([1, 2]))
    model = TinyTreeModel()
    result = ARTreeScorer(model, 2).score_tree(torch.tensor([1, 2]), tree)
    assert model.forward_calls == 1
    assert result["verifier_forward_calls"] == 1
    expected = sum(len(node.candidate_set) for node in tree.nodes.values())
    assert result["tree_candidate_blocks_scored"] == expected
    for node in tree.nodes.values():
        if not node.candidate_set:
            continue
        assert all(candidate.verifier_log_score is not None for candidate in node.candidate_set)
        assert abs(sum(candidate.p_normalized for candidate in node.candidate_set) - 1.0) < 1e-6
