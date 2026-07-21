from __future__ import annotations

from dataclasses import dataclass

from .types import DraftTreeNode


@dataclass
class DraftTree:
    nodes: dict[int, DraftTreeNode]
    root_id: int = 0

    @property
    def root(self) -> DraftTreeNode:
        return self.nodes[self.root_id]

    def child_for_candidate(self, node: DraftTreeNode, candidate_index: int) -> DraftTreeNode | None:
        block = node.candidate_set[candidate_index].token_ids
        for child_id in node.children:
            child = self.nodes[child_id]
            if child.block_token_ids == block:
                return child
        return None

    def widths(self) -> list[int]:
        if not self.nodes:
            return []
        max_depth = max(n.depth for n in self.nodes.values())
        return [sum(n.depth == d for n in self.nodes.values()) for d in range(1, max_depth + 1)]

