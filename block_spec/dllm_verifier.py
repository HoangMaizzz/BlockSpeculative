from __future__ import annotations

import math
import gc

import torch

from .candidate_search import heap_topk_joint, score_block_from_marginals
from .distributions import normalize_log_scores
from .verifier_candidate_search import VerifierBeam, _node_prefixes


class FastDLLMVerifier:
    """Block scorer/searcher for a Fast-dLLM verifier.

    At every expanded tree node the verifier sees the node's complete ancestor
    prefix followed by one fully masked native block.  The first ``block_size``
    masked-position marginals define

        p(block | prefix) = product_i p_i(token_i | masked block, prefix).

    This deliberately mirrors ``FastDLLMv2Adapter``; it is not AR teacher
    forcing and does not use AR beam search.
    """

    def __init__(
        self,
        adapter,
        topk: int = 5,
        per_position_topk: int = 8,
        node_batch_size: int = 4,
    ):
        self.adapter = adapter
        self.topk = int(topk)
        self.per_position_topk = int(per_position_topk)
        self.node_batch_size = max(1, int(node_batch_size))
        self.batch_size_used = self.node_batch_size
        if self.topk < 1 or self.per_position_topk < 1:
            raise ValueError("top-k values must be positive")
        self._tables: dict[int, torch.Tensor] = {}

    def clear_cache(self) -> None:
        self._tables.clear()

    def _table(self, node_id: int, prefix: torch.LongTensor) -> torch.Tensor:
        table = self._tables.get(node_id)
        if table is None:
            table = self.adapter.marginal_log_probs(prefix)
            self._tables[node_id] = table.to(dtype=torch.float16).contiguous()
        return self._tables[node_id]

    def _populate_tables(
        self, prefixes: dict[int, torch.LongTensor]
    ) -> int:
        """Batch all missing node prefixes by length (equivalently tree depth)."""
        groups: dict[int, list[tuple[int, torch.LongTensor]]] = {}
        for node_id, prefix in prefixes.items():
            if node_id in self._tables:
                continue
            groups.setdefault(int(prefix.numel()), []).append((node_id, prefix))
        forward_calls = 0
        batch_method = getattr(self.adapter, "marginal_log_probs_batch", None)
        batch_size = self.node_batch_size
        for items in groups.values():
            cursor = 0
            while cursor < len(items):
                chunk = items[cursor : cursor + batch_size]
                try:
                    if batch_method is None:
                        tables = torch.stack(
                            [
                                self.adapter.marginal_log_probs(prefix)
                                for _, prefix in chunk
                            ]
                        )
                        forward_calls += len(chunk)
                    else:
                        tables = batch_method([prefix for _, prefix in chunk])
                        forward_calls += 1
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise RuntimeError(
                            "Fast-dLLM verifier OOM at node batch size 1"
                        )
                    batch_size = max(1, batch_size // 2)
                    self.batch_size_used = batch_size
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(
                        "CUDA OOM: retrying Fast-dLLM verifier with "
                        f"node batch size {batch_size}",
                        flush=True,
                    )
                    continue
                for row, (node_id, _) in enumerate(chunk):
                    self._tables[node_id] = tables[row].to(
                        dtype=torch.float16
                    ).contiguous()
                cursor += len(chunk)
                del tables
        self.batch_size_used = min(self.batch_size_used, batch_size)
        return forward_calls

    @torch.inference_mode()
    def search_tree(self, prefix_ids: torch.LongTensor, tree):
        prefixes = _node_prefixes(prefix_ids, tree)
        searchable = {
            node.node_id: prefixes[node.node_id]
            for node in tree.nodes.values()
            if node.candidate_set
        }
        forward_calls = self._populate_tables(searchable)
        result: dict[int, list[VerifierBeam]] = {}
        searched = 0
        for node in sorted(tree.nodes.values(), key=lambda item: item.node_id):
            if not node.candidate_set:
                continue
            table = self._table(node.node_id, prefixes[node.node_id])
            found = heap_topk_joint(table, self.per_position_topk, self.topk)
            result[node.node_id] = [
                VerifierBeam(
                    token_ids=candidate.token_ids,
                    token_logprobs=candidate.drafter_token_logprobs,
                    log_score=candidate.drafter_log_score,
                )
                for candidate in found.candidates
            ]
            searched += 1
        return result, {
            "backend": "fast_dllm_masked_marginals",
            "nodes_searched": searched,
            "block_size": self.adapter.block_size,
            "verifier_block_topk": self.topk,
            "per_position_topk": self.per_position_topk,
            "verifier_forward_calls": forward_calls,
            "node_batch_size_used": self.batch_size_used,
            "batching": "equal-prefix-length nodes grouped then chunked",
            "joint_probability": "product_of_masked_position_marginals",
        }

    @torch.inference_mode()
    def score_tree(self, prefix_ids: torch.LongTensor, tree) -> dict:
        prefixes = _node_prefixes(prefix_ids, tree)
        scorable = {
            node.node_id: prefixes[node.node_id]
            for node in tree.nodes.values()
            if node.candidate_set
        }
        scoring_forward_calls = self._populate_tables(scorable)
        node_statistics = []
        candidate_blocks = 0
        projected_logit_count = 0
        for node in sorted(tree.nodes.values(), key=lambda item: item.node_id):
            if not node.candidate_set:
                continue
            table = self._table(node.node_id, prefixes[node.node_id])
            for candidate in node.candidate_set:
                token_logs, block_log = score_block_from_marginals(
                    table, candidate.token_ids
                )
                candidate.verifier_token_logprobs = token_logs
                candidate.verifier_log_score = block_log
                projected_logit_count += len(candidate.token_ids)
            q_logs = torch.tensor(
                [candidate.drafter_log_score for candidate in node.candidate_set],
                dtype=torch.float64,
            )
            p_logs = torch.tensor(
                [candidate.verifier_log_score for candidate in node.candidate_set],
                dtype=torch.float64,
            )
            q_probs = normalize_log_scores(q_logs)
            p_probs = normalize_log_scores(p_logs)
            for index, candidate in enumerate(node.candidate_set):
                candidate.q_normalized = float(q_probs[index])
                candidate.p_normalized = float(p_probs[index])
            node_statistics.append(
                {
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "candidate_count": len(node.candidate_set),
                    "raw_candidate_mass_q": sum(math.exp(float(x)) for x in q_logs),
                    "raw_candidate_mass_p": sum(math.exp(float(x)) for x in p_logs),
                    "sampling_probability_space": "retained_union",
                    "coverage_probability_space": "full_verifier_factorized_block_space",
                }
            )
            candidate_blocks += len(node.candidate_set)
        # If search_tree ran first, every table was cached and scoring needed no
        # additional verifier forwards.  In drafter_topk mode each node is run here.
        total_tables = len(self._tables)
        return {
            "backend": "fast_dllm_masked_marginals",
            "verifier_forward_calls": scoring_forward_calls,
            "cached_node_tables": total_tables,
            "node_batch_size_used": self.batch_size_used,
            "tree_candidate_blocks_scored": candidate_blocks,
            "tree_candidate_tokens_scored": projected_logit_count,
            "probability_space": "factorized_masked_block_marginals",
            "probability_normalization": "full_vocab_per_masked_position",
            "coverage_is_full_vocab_exact": True,
            "coverage_seconds": 0.0,
            "full_vocabulary_logits_materialized": True,
            "joint_probability": "product_of_masked_position_marginals",
            "node_statistics": node_statistics,
        }
