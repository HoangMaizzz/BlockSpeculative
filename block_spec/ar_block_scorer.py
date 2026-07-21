from __future__ import annotations

import gc

import torch

from .types import ARBlockScoreResult


class ARBlockScorer:
    def __init__(self, model, candidate_batch_size: int = 16):
        self.model = model
        self.candidate_batch_size = candidate_batch_size

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def score_blocks(self, prefix_ids, candidate_blocks, past_key_values=None) -> ARBlockScoreResult:
        prefix = prefix_ids.reshape(1, -1).long()
        blocks = candidate_blocks.long()
        if prefix.shape[1] == 0:
            raise ValueError("AR scoring requires a non-empty prefix")
        batch_size = min(self.candidate_batch_size, blocks.shape[0])
        while True:
            try:
                token_parts, final_parts = [], []
                for start in range(0, blocks.shape[0], batch_size):
                    chunk = blocks[start : start + batch_size].to(self.device)
                    repeated = prefix.to(self.device).expand(chunk.shape[0], -1)
                    ids = torch.cat((repeated, chunk), dim=1)
                    output = self.model(input_ids=ids, use_cache=False)
                    logits = output.logits.float()
                    begin = prefix.shape[1] - 1
                    prediction_logits = logits[:, begin : begin + chunk.shape[1], :]
                    lps = torch.log_softmax(prediction_logits, dim=-1)
                    selected = lps.gather(-1, chunk.unsqueeze(-1)).squeeze(-1)
                    token_parts.append(selected.cpu())
                    final_parts.append(logits[:, -1, :].cpu())
                    del ids, output, logits, lps
                tokens = torch.cat(token_parts)
                return ARBlockScoreResult(tokens, tokens.sum(-1), torch.cat(final_parts), batch_size)
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise RuntimeError("Verifier OOM even with candidate batch size 1")
                batch_size = max(1, batch_size // 2)
                print(f"CUDA OOM: retrying verifier candidates with batch size {batch_size}")
                gc.collect()
                torch.cuda.empty_cache()

    def score_blocks_slow(self, prefix_ids, candidate_blocks) -> ARBlockScoreResult:
        old = self.candidate_batch_size
        try:
            self.candidate_batch_size = 1
            return self.score_blocks(prefix_ids, candidate_blocks)
        finally:
            self.candidate_batch_size = old
