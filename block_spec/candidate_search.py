from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import torch

from .types import BlockCandidate


@dataclass
class CandidateSearchResult:
    candidates: list[BlockCandidate]
    retained_mass_per_position: tuple[float, ...]
    candidate_mass: float


def heap_topk_joint(
    log_probs: torch.Tensor,
    per_position_topk: int,
    num_block_candidates: int,
) -> CandidateSearchResult:
    """Best-first search of the Cartesian product of sorted marginal top-k lists."""
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [block_size, vocab_size]")
    if per_position_topk < 1 or num_block_candidates < 1:
        raise ValueError("top-k values must be positive")
    k = min(per_position_topk, log_probs.shape[-1])
    values, ids = torch.topk(log_probs.float(), k=k, dim=-1, largest=True, sorted=True)
    masses = torch.exp(values).sum(-1)
    block_size = log_probs.shape[0]

    def score(ranks: tuple[int, ...]) -> float:
        return sum(float(values[pos, rank]) for pos, rank in enumerate(ranks))

    origin = (0,) * block_size
    heap: list[tuple[float, tuple[int, ...]]] = [(-score(origin), origin)]
    visited = {origin}
    candidates: list[BlockCandidate] = []
    seen_blocks: set[tuple[int, ...]] = set()
    while heap and len(candidates) < num_block_candidates:
        neg_score, ranks = heapq.heappop(heap)
        token_ids = tuple(int(ids[pos, rank]) for pos, rank in enumerate(ranks))
        token_lps = tuple(float(values[pos, rank]) for pos, rank in enumerate(ranks))
        if token_ids not in seen_blocks:
            seen_blocks.add(token_ids)
            candidates.append(BlockCandidate(token_ids, token_lps, -neg_score, rank=len(candidates)))
        for pos in range(block_size):
            if ranks[pos] + 1 >= k:
                continue
            nxt = list(ranks)
            nxt[pos] += 1
            nxt_t = tuple(nxt)
            if nxt_t not in visited:
                visited.add(nxt_t)
                heapq.heappush(heap, (-score(nxt_t), nxt_t))
    candidates.sort(key=lambda c: (-c.drafter_log_score, c.token_ids))
    for rank, candidate in enumerate(candidates):
        candidate.rank = rank
    candidate_mass = sum(math.exp(c.drafter_log_score) for c in candidates)
    return CandidateSearchResult(candidates, tuple(float(x) for x in masses), candidate_mass)


def brute_force_topk(log_probs: torch.Tensor, per_position_topk: int, n: int) -> list[tuple[tuple[int, ...], float]]:
    import itertools
    k = min(per_position_topk, log_probs.shape[-1])
    values, ids = torch.topk(log_probs.float(), k=k, dim=-1, sorted=True)
    rows = []
    for ranks in itertools.product(range(k), repeat=log_probs.shape[0]):
        toks = tuple(int(ids[i, r]) for i, r in enumerate(ranks))
        rows.append((toks, sum(float(values[i, r]) for i, r in enumerate(ranks))))
    return sorted(rows, key=lambda x: (-x[1], x[0]))[:n]

