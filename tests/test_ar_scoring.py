from types import SimpleNamespace

import torch

from block_spec.ar_block_scorer import ARBlockScorer


class TinyCausal(torch.nn.Module):
    def __init__(self, vocab=11):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 8)
        self.head = torch.nn.Linear(8, vocab, bias=False)
    def get_input_embeddings(self): return self.embed
    def forward(self, input_ids, use_cache=False):
        # Position-independent is sufficient to validate gather/shift/batching.
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def test_batched_equals_single_and_sums():
    torch.manual_seed(1)
    model = TinyCausal()
    prefix = torch.tensor([[1, 2, 3]])
    blocks = torch.tensor([[4, 5], [6, 7], [8, 9]])
    scorer = ARBlockScorer(model, 3)
    batched = scorer.score_blocks(prefix, blocks)
    slow = scorer.score_blocks_slow(prefix, blocks)
    assert torch.allclose(batched.token_logprobs, slow.token_logprobs)
    assert torch.allclose(batched.block_logprobs, batched.token_logprobs.sum(-1))
