from types import SimpleNamespace

import torch

from block_spec.ar_block_scorer import ARBlockScorer
from block_spec.config import GenerationConfig
from block_spec.decoder import BlockSpeculativeDecoder
from block_spec.drafter_adapter import DraftCandidates
from block_spec.tree_builder import AsymmetricTreeBuilder
from block_spec.types import BlockCandidate


class TinyTokenizer:
    eos_token_id = 0
    def __call__(self, text, return_tensors=None): return {"input_ids": torch.tensor([[1, 2]])}
    def decode(self, ids, skip_special_tokens=True): return " ".join(map(str, ids))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embed = torch.nn.Embedding(8, 4); self.head = torch.nn.Linear(4, 8)
    def get_input_embeddings(self): return self.embed
    def forward(self, input_ids, use_cache=False): return SimpleNamespace(logits=self.head(self.embed(input_ids)))


class OneCandidateDrafter:
    block_size = 2
    def propose(self, prefix):
        c = BlockCandidate((3, 4), (-0.1, -0.2), -0.3)
        return DraftCandidates([c], (0.9, 0.9), 0.74)


def test_end_to_end_dummy_generation():
    drafter = OneCandidateDrafter()
    tree = AsymmetricTreeBuilder(drafter, [1], [1], 0, 4, 8)
    cfg = {
        "seed": 1,
        "sampling": {"proposal_mode": "top1_q", "temperature": 0.0, "top_p": 1.0, "top_k": None,
                     "enable_bonus_token": False},
        "residual": {"epsilon": 1e-12, "mass_warning_threshold": 0.0},
        "generation": {"max_new_tokens": 2, "fallback_mode": "target_one_token"},
    }
    decoder = BlockSpeculativeDecoder(drafter, ARBlockScorer(TinyModel(), 2), TinyTokenizer(), tree, cfg)
    result = decoder.generate("hello", GenerationConfig(max_new_tokens=2))
    assert result.token_ids == [3, 4]
    assert result.generated_token_count == 2
    assert len(result.rounds) == 1


def test_verify_prebuilt_tree_does_not_rebuild_and_commits_existing_path():
    drafter = OneCandidateDrafter()
    builder = AsymmetricTreeBuilder(drafter, [1, 1], [1, 1], 0, 4, 8)
    prefix = torch.tensor([1, 2])
    tree = builder.build(prefix)
    cfg = {
        "seed": 1,
        "sampling": {"proposal_mode": "top1_q", "temperature": 0.0, "top_p": 1.0, "top_k": None},
        "residual": {"epsilon": 1e-12, "mass_warning_threshold": 0.0},
        "logging": {"print_verification_trace": False},
    }
    decoder = BlockSpeculativeDecoder(
        drafter, ARBlockScorer(TinyModel(), 2), TinyTokenizer(), builder, cfg
    )
    result = decoder.verify_prebuilt_tree(tree, prefix, max_blocks=2)
    assert result["blocks_committed"] == 2
    assert result["visited_node_ids"] == [0, 1]
    assert result["committed_token_ids"] == [3, 4, 3, 4]
    assert result["stop_reason"] == "reached_final_expanded_depth"


def test_traverse_precomputed_tree_uses_no_model_calls():
    drafter = OneCandidateDrafter()
    builder = AsymmetricTreeBuilder(drafter, [1, 1], [1, 1], 0, 4, 8)
    prefix = torch.tensor([1, 2])
    tree = builder.build(prefix)
    for node in tree.nodes.values():
        for candidate in node.candidate_set:
            candidate.verifier_token_logprobs = (-0.1, -0.2)
            candidate.verifier_log_score = -0.3
            candidate.q_normalized = 1.0
            candidate.p_normalized = 1.0
    cfg = {
        "seed": 1,
        "sampling": {"proposal_mode": "top1_q"},
        "residual": {"epsilon": 1e-12},
        "logging": {"print_verification_trace": False},
    }
    decoder = BlockSpeculativeDecoder(drafter, None, TinyTokenizer(), builder, cfg)
    result = decoder.traverse_precomputed_tree(tree, prefix, max_blocks=2)
    assert result["blocks_committed"] == 2
    assert result["model_calls_during_traversal"] == 0
    assert result["committed_token_ids"] == [3, 4, 3, 4]
