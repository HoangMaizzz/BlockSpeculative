from __future__ import annotations


def format_tree(tree, tokenizer=None) -> str:
    """Return a compact deterministic text view suitable for Colab logs."""
    lines = []
    for node in sorted(tree.nodes.values(), key=lambda n: (n.depth, n.node_id)):
        block = node.block_token_ids
        text = tokenizer.decode(block) if tokenizer is not None and block else "ROOT"
        lines.append(
            f"{'  ' * node.depth}[{node.node_id}] {text!r} "
            f"score={node.cumulative_drafter_log_score:.4f} candidates={len(node.candidate_set)} "
            f"children={len(node.children)}"
        )
    return "\n".join(lines)
