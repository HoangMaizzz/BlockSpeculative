from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .distributions import normalize_log_scores


@dataclass(frozen=True)
class TreeCandidateEdge:
    parent_node_id: int
    candidate_index: int
    start: int
    end: int
    context_last_position: int


@dataclass
class TreeAttentionLayout:
    input_ids: torch.LongTensor
    position_ids: torch.LongTensor
    allowed_attention: torch.BoolTensor
    edges: list[TreeCandidateEdge]


def build_tree_attention_layout(prefix_ids: torch.LongTensor, tree) -> TreeAttentionLayout:
    """Flatten all local candidate blocks and build ancestor-only attention.

    Every candidate block is represented, including unexpanded residual leaves.
    Expanded child nodes inherit the span of their incoming candidate edge as an
    ancestor. Sibling and cousin spans are never mutually visible.
    """
    prefix = prefix_ids.reshape(-1).long().cpu()
    if prefix.numel() == 0:
        raise ValueError("Tree AR scoring requires a non-empty prefix")
    flat_ids = prefix.tolist()
    logical_positions = list(range(prefix.numel()))
    edges: list[TreeCandidateEdge] = []
    # For each expanded node, store all incoming block spans on its unique path.
    ancestor_spans: dict[int, list[tuple[int, int]]] = {tree.root_id: []}
    context_last: dict[int, int] = {tree.root_id: int(prefix.numel()) - 1}
    edge_ancestors: list[list[tuple[int, int]]] = []
    ordered_nodes = sorted(tree.nodes.values(), key=lambda node: (node.depth, node.node_id))
    for node in ordered_nodes:
        if not node.candidate_set:
            continue
        if node.node_id not in ancestor_spans:
            raise RuntimeError(f"Node {node.node_id} has no mapped ancestor path")
        for candidate_index, candidate in enumerate(node.candidate_set):
            start = len(flat_ids)
            flat_ids.extend(candidate.token_ids)
            end = len(flat_ids)
            logical_start = int(prefix.numel()) + node.depth * len(candidate.token_ids)
            logical_positions.extend(range(logical_start, logical_start + len(candidate.token_ids)))
            edges.append(
                TreeCandidateEdge(
                    parent_node_id=node.node_id,
                    candidate_index=candidate_index,
                    start=start,
                    end=end,
                    context_last_position=context_last[node.node_id],
                )
            )
            edge_ancestors.append(list(ancestor_spans[node.node_id]))
            child = tree.child_for_candidate(node, candidate_index)
            if child is not None:
                ancestor_spans[child.node_id] = [*ancestor_spans[node.node_id], (start, end)]
                context_last[child.node_id] = end - 1
    total_length = len(flat_ids)
    allowed = torch.zeros((total_length, total_length), dtype=torch.bool)
    prefix_length = int(prefix.numel())
    # Preserve ordinary causal attention inside the prompt.
    for query in range(prefix_length):
        allowed[query, : query + 1] = True
    # A tree token sees the prompt, complete ancestor blocks, and its own block causally.
    for edge, ancestors in zip(edges, edge_ancestors):
        for offset, query in enumerate(range(edge.start, edge.end)):
            allowed[query, :prefix_length] = True
            for ancestor_start, ancestor_end in ancestors:
                allowed[query, ancestor_start:ancestor_end] = True
            allowed[query, edge.start : edge.start + offset + 1] = True
    return TreeAttentionLayout(
        input_ids=torch.tensor(flat_ids, dtype=torch.long).unsqueeze(0),
        position_ids=torch.tensor(logical_positions, dtype=torch.long).unsqueeze(0),
        allowed_attention=allowed,
        edges=edges,
    )


class ARTreeScorer:
    """Score every candidate block in a prebuilt tree with one AR forward."""

    def __init__(self, model, logsumexp_row_chunk_size: int = 32):
        self.model = model
        self.logsumexp_row_chunk_size = max(1, int(logsumexp_row_chunk_size))

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def score_tree(self, prefix_ids: torch.LongTensor, tree) -> dict:
        layout = build_tree_attention_layout(prefix_ids, tree)
        device = self.device
        model_dtype = self.model.get_input_embeddings().weight.dtype
        minimum = torch.finfo(model_dtype).min
        additive_mask = torch.full(
            layout.allowed_attention.shape,
            minimum,
            dtype=model_dtype,
            device=device,
        )
        additive_mask.masked_fill_(layout.allowed_attention.to(device), 0.0)
        additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)
        output = self.model(
            input_ids=layout.input_ids.to(device),
            attention_mask=additive_mask,
            position_ids=layout.position_ids.to(device),
            use_cache=False,
        )
        logits = output.logits[0]
        prediction_positions: list[int] = []
        target_token_ids: list[int] = []
        edge_slices: list[tuple[int, int]] = []
        cursor = 0
        for edge in layout.edges:
            tokens = layout.input_ids[0, edge.start:edge.end].tolist()
            positions = [edge.context_last_position, *range(edge.start, edge.end - 1)]
            prediction_positions.extend(positions)
            target_token_ids.extend(tokens)
            edge_slices.append((cursor, cursor + len(tokens)))
            cursor += len(tokens)
        prediction_tensor = torch.tensor(prediction_positions, dtype=torch.long, device=logits.device)
        target_tensor = torch.tensor(target_token_ids, dtype=torch.long, device=logits.device)
        unique_positions, inverse = torch.unique(prediction_tensor, sorted=True, return_inverse=True)
        denominators = torch.empty(unique_positions.numel(), dtype=torch.float32, device=logits.device)
        for start in range(0, unique_positions.numel(), self.logsumexp_row_chunk_size):
            positions = unique_positions[start : start + self.logsumexp_row_chunk_size]
            denominators[start : start + positions.numel()] = torch.logsumexp(
                logits.index_select(0, positions).float(), dim=-1
            )
        target_logits = logits[prediction_tensor, target_tensor].float()
        token_logprobs = (target_logits - denominators[inverse]).cpu()
        for edge, (start, end) in zip(layout.edges, edge_slices):
            candidate = tree.nodes[edge.parent_node_id].candidate_set[edge.candidate_index]
            values = token_logprobs[start:end]
            candidate.verifier_token_logprobs = tuple(float(value) for value in values)
            candidate.verifier_log_score = float(values.sum())
        node_statistics = []
        for node in sorted(tree.nodes.values(), key=lambda item: item.node_id):
            if not node.candidate_set:
                continue
            q_logs = torch.tensor([c.drafter_log_score for c in node.candidate_set], dtype=torch.float64)
            p_logs = torch.tensor([c.verifier_log_score for c in node.candidate_set], dtype=torch.float64)
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
                    "raw_candidate_mass_q": sum(math.exp(float(value)) for value in q_logs),
                    "raw_candidate_mass_p": sum(math.exp(float(value)) for value in p_logs),
                }
            )
        result = {
            "verifier_forward_calls": 1,
            "flattened_sequence_length": int(layout.input_ids.shape[1]),
            "tree_candidate_blocks_scored": len(layout.edges),
            "tree_candidate_tokens_scored": sum(edge.end - edge.start for edge in layout.edges),
            "attention_mask_shape": list(additive_mask.shape),
            "node_statistics": node_statistics,
        }
        del output, logits, additive_mask, denominators, target_logits
        return result
