from __future__ import annotations

import math
from collections import Counter

import torch

from .tree import DraftTree
from .types import BlockCandidate, DraftTreeNode


class AsymmetricTreeBuilder:
    def __init__(
        self,
        drafter,
        width_schedule=(5, 20, 10, 5, 2, 1),
        parent_cap_schedule=(5, 5, 3, 2, 1, 1),
        depth2_min_children_per_parent=2,
        max_tree_nodes=64,
        max_tree_tokens=320,
        confidence_mode="cumulative_log_score",
        hybrid_mean_weight=0.5,
        hybrid_min_weight=0.5,
    ):
        self.drafter = drafter
        self.width_schedule = list(width_schedule)
        self.parent_caps = list(parent_cap_schedule)
        self.depth2_quota = depth2_min_children_per_parent
        self.max_tree_nodes = max_tree_nodes
        self.max_tree_tokens = max_tree_tokens
        self.confidence_mode = confidence_mode
        self.hybrid_mean_weight = hybrid_mean_weight
        self.hybrid_min_weight = hybrid_min_weight

    def _confidence(self, candidate: BlockCandidate, cumulative: float) -> float:
        if self.confidence_mode == "cumulative_log_score":
            return cumulative
        if self.confidence_mode == "mean_token_logprob":
            return sum(candidate.drafter_token_logprobs) / len(candidate.drafter_token_logprobs)
        if self.confidence_mode == "min_token_logprob":
            return min(candidate.drafter_token_logprobs)
        if self.confidence_mode == "hybrid":
            mean = sum(candidate.drafter_token_logprobs) / len(candidate.drafter_token_logprobs)
            return self.hybrid_mean_weight * mean + self.hybrid_min_weight * min(candidate.drafter_token_logprobs)
        raise ValueError(f"Unknown confidence mode: {self.confidence_mode}")

    @torch.inference_mode()
    def build(self, prefix_ids: torch.LongTensor) -> DraftTree:
        first = self.drafter.propose(prefix_ids)
        root = DraftTreeNode(0, None, 0, None, 0.0, 0.0, 0.0, first.candidates)
        nodes = {0: root}
        active = [root]
        next_id = 1
        prefix_list = prefix_ids.reshape(-1).tolist()
        paths: dict[int, list[int]] = {0: prefix_list}
        for depth, target_width in enumerate(self.width_schedule, start=1):
            eligible: list[tuple[float, int, int, BlockCandidate]] = []
            for parent in active:
                cap = self.parent_caps[min(depth - 1, len(self.parent_caps) - 1)]
                ranked = sorted(enumerate(parent.candidate_set), key=lambda x: (-x[1].drafter_log_score, x[1].token_ids))
                for idx, candidate in ranked[:cap]:
                    cumulative = parent.cumulative_drafter_log_score + candidate.drafter_log_score
                    eligible.append((self._confidence(candidate, cumulative), parent.node_id, idx, candidate))
            if not eligible:
                break
            eligible.sort(key=lambda x: (-x[0], x[1], x[2], x[3].token_ids))
            selected = []
            if depth == 2 and self.depth2_quota:
                counts = Counter()
                for item in eligible:
                    if counts[item[1]] < self.depth2_quota and len(selected) < target_width:
                        selected.append(item)
                        counts[item[1]] += 1
                chosen = {(x[1], x[2]) for x in selected}
                selected.extend(x for x in eligible if (x[1], x[2]) not in chosen and len(selected) < target_width)
            else:
                selected = eligible[:target_width]
            budget_nodes = self.max_tree_nodes - len(nodes)
            budget_tokens = (self.max_tree_tokens - (len(nodes) - 1) * self.drafter.block_size) // self.drafter.block_size
            selected = selected[: max(0, min(budget_nodes, budget_tokens))]
            if not selected:
                break
            new_active = []
            for confidence, parent_id, candidate_idx, candidate in selected:
                parent = nodes[parent_id]
                path = paths[parent_id] + list(candidate.token_ids)
                cumulative = parent.cumulative_drafter_log_score + candidate.drafter_log_score
                child = DraftTreeNode(
                    next_id, parent_id, depth, candidate.token_ids,
                    candidate.drafter_log_score, cumulative, confidence,
                )
                if depth < len(self.width_schedule):
                    proposed = self.drafter.propose(torch.tensor(path, dtype=torch.long))
                    child.candidate_set = proposed.candidates
                nodes[next_id] = child
                paths[next_id] = path
                parent.expanded_candidate_indices.append(candidate_idx)
                parent.children.append(next_id)
                parent.is_leaf = False
                new_active.append(child)
                next_id += 1
            active = new_active
        return DraftTree(nodes)

