from __future__ import annotations

from dataclasses import dataclass
import gc

import torch

from .candidate_search import score_block_from_marginals
from .types import BlockCandidate


@dataclass(frozen=True)
class VerifierBeam:
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    log_score: float


def _node_prefixes(prefix_ids: torch.LongTensor, tree) -> dict[int, torch.LongTensor]:
    paths = {tree.root_id: prefix_ids.reshape(-1).long().cpu()}
    for node in sorted(tree.nodes.values(), key=lambda item: (item.depth, item.node_id)):
        if node.node_id == tree.root_id:
            continue
        parent_prefix = paths[node.parent_id]
        paths[node.node_id] = torch.cat(
            (parent_prefix, torch.tensor(node.block_token_ids, dtype=torch.long))
        )
    return paths


class ARVerifierTopKBlockSearch:
    """Batched AR beam approximation of verifier Top-K blocks for many nodes."""

    def __init__(self, model, block_size: int, topk: int = 5, batch_size: int = 16):
        self.model = model
        self.block_size = int(block_size)
        self.topk = int(topk)
        self.batch_size = max(1, int(batch_size))
        if self.block_size < 1 or self.topk < 1:
            raise ValueError("block_size and topk must be positive")

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def _last_topk(
        self, sequences: list[torch.LongTensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([sequence.numel() for sequence in sequences], dtype=torch.long)
        max_length = int(lengths.max())
        ids = torch.zeros((len(sequences), max_length), dtype=torch.long, device=self.device)
        attention = torch.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            length = sequence.numel()
            ids[row, :length] = sequence.to(self.device)
            attention[row, :length] = 1
        position_ids = attention.cumsum(-1) - 1
        position_ids.masked_fill_(attention == 0, 0)
        base_model = getattr(self.model, "model", None)
        lm_head = getattr(self.model, "lm_head", None)
        if base_model is None or lm_head is None:
            output = self.model(
                input_ids=ids,
                attention_mask=attention,
                position_ids=position_ids,
                use_cache=False,
            )
            rows = torch.arange(len(sequences), device=output.logits.device)
            logits = output.logits[rows, lengths.to(output.logits.device) - 1]
        else:
            output = base_model(
                input_ids=ids,
                attention_mask=attention,
                position_ids=position_ids,
                use_cache=False,
            )
            rows = torch.arange(len(sequences), device=output.last_hidden_state.device)
            last_hidden = output.last_hidden_state[
                rows, lengths.to(output.last_hidden_state.device) - 1
            ]
            logits = lm_head(last_hidden)
        values, token_ids = torch.topk(
            logits, k=min(self.topk, logits.shape[-1]), dim=-1, sorted=True
        )
        result = (values.float().cpu(), token_ids.cpu())
        del ids, attention, position_ids, output, logits
        return result

    def _expand_stage(
        self,
        prefixes: dict[int, torch.LongTensor],
        beams: dict[int, list[VerifierBeam]],
        batch_size: int,
    ) -> dict[int, list[VerifierBeam]]:
        rows: list[tuple[int, VerifierBeam, torch.LongTensor]] = []
        for node_id, node_beams in beams.items():
            for beam in node_beams:
                suffix = torch.tensor(beam.token_ids, dtype=torch.long)
                rows.append((node_id, beam, torch.cat((prefixes[node_id], suffix))))
        expanded: dict[int, list[VerifierBeam]] = {node_id: [] for node_id in beams}
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            values, token_ids = self._last_topk([item[2] for item in chunk])
            # The retained verifier Top-K token set is treated as 100% mass.
            local_log_probs = torch.log_softmax(values, dim=-1)
            for row_index, (node_id, beam, _) in enumerate(chunk):
                for rank in range(token_ids.shape[1]):
                    token_id = int(token_ids[row_index, rank])
                    token_lp = float(local_log_probs[row_index, rank])
                    expanded[node_id].append(
                        VerifierBeam(
                            (*beam.token_ids, token_id),
                            (*beam.token_logprobs, token_lp),
                            beam.log_score + token_lp,
                        )
                    )
            del values, token_ids, local_log_probs
        result: dict[int, list[VerifierBeam]] = {}
        for node_id, candidates in expanded.items():
            unique: dict[tuple[int, ...], VerifierBeam] = {}
            for candidate in sorted(candidates, key=lambda item: (-item.log_score, item.token_ids)):
                unique.setdefault(candidate.token_ids, candidate)
                if len(unique) == self.topk:
                    break
            result[node_id] = list(unique.values())
        return result

    def search_tree(self, prefix_ids: torch.LongTensor, tree) -> tuple[dict[int, list[VerifierBeam]], dict]:
        prefixes = _node_prefixes(prefix_ids, tree)
        searchable = {
            node.node_id
            for node in tree.nodes.values()
            if node.candidate_set and node.drafter_marginal_logprobs is not None
        }
        prefixes = {node_id: prefixes[node_id] for node_id in sorted(searchable)}
        beams = {node_id: [VerifierBeam((), (), 0.0)] for node_id in prefixes}
        batch_size = self.batch_size
        forward_calls = 0
        for _ in range(self.block_size):
            while True:
                try:
                    row_count = sum(len(node_beams) for node_beams in beams.values())
                    beams = self._expand_stage(prefixes, beams, batch_size)
                    forward_calls += (row_count + batch_size - 1) // batch_size
                    break
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise RuntimeError("Verifier Top-K beam search OOM at batch size 1")
                    batch_size = max(1, batch_size // 2)
                    print(f"CUDA OOM: retrying verifier beam search with batch size {batch_size}")
                    gc.collect()
                    torch.cuda.empty_cache()
        return beams, {
            "nodes_searched": len(prefixes),
            "block_size": self.block_size,
            "verifier_block_topk": self.topk,
            "batch_size_used": batch_size,
            "verifier_forward_calls": forward_calls,
        }


def merge_verifier_topk_into_tree(tree, verifier_beams: dict[int, list[VerifierBeam]]) -> dict:
    added = 0
    duplicates = 0
    for node_id, beams in verifier_beams.items():
        node = tree.nodes[node_id]
        table = node.drafter_marginal_logprobs
        if table is None:
            raise RuntimeError(f"Node {node_id} has no saved drafter marginal table")
        by_tokens = {candidate.token_ids: candidate for candidate in node.candidate_set}
        for beam in beams:
            existing = by_tokens.get(beam.token_ids)
            if existing is not None:
                existing.candidate_source = "drafter+verifier"
                duplicates += 1
                continue
            token_q, joint_q = score_block_from_marginals(table, beam.token_ids)
            candidate = BlockCandidate(
                token_ids=beam.token_ids,
                drafter_token_logprobs=token_q,
                drafter_log_score=joint_q,
                verifier_token_logprobs=beam.token_logprobs,
                verifier_log_score=beam.log_score,
                candidate_source="verifier",
            )
            node.candidate_set.append(candidate)
            by_tokens[candidate.token_ids] = candidate
            added += 1
    return {"verifier_only_blocks_added": added, "duplicate_blocks_merged": duplicates}
