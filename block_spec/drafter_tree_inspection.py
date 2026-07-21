from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .distributions import normalize_log_scores


def _decode_token(tokenizer, token_id: int) -> tuple[str, str]:
    token_piece = tokenizer.convert_ids_to_tokens(token_id) if hasattr(tokenizer, "convert_ids_to_tokens") else str(token_id)
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    return str(token_piece), decoded.replace("\n", "\\n")


def build_first_step_tree_report(
    *,
    prompt: str,
    prompt_token_count: int,
    draft,
    tokenizer,
    block_size: int,
    per_position_topk: int,
) -> dict[str, Any]:
    """Build a serializable depth-1 report from one drafter proposal call."""
    if not draft.candidates:
        raise RuntimeError("Drafter returned no block candidates")
    log_scores = torch.tensor([candidate.drafter_log_score for candidate in draft.candidates], dtype=torch.float64)
    normalized = normalize_log_scores(log_scores)
    blocks = []
    for index, candidate in enumerate(draft.candidates):
        token_rows = []
        for position, (token_id, log_probability) in enumerate(
            zip(candidate.token_ids, candidate.drafter_token_logprobs), start=1
        ):
            token_piece, decoded = _decode_token(tokenizer, token_id)
            token_rows.append(
                {
                    "position": position,
                    "token_id": int(token_id),
                    "token_piece": token_piece,
                    "decoded": decoded,
                    "log_probability": float(log_probability),
                    "probability": math.exp(float(log_probability)),
                }
            )
        raw_joint = math.exp(float(candidate.drafter_log_score))
        product = math.prod(row["probability"] for row in token_rows)
        blocks.append(
            {
                "rank": index + 1,
                "token_ids": list(candidate.token_ids),
                "decoded_block": tokenizer.decode(list(candidate.token_ids), skip_special_tokens=False).replace("\n", "\\n"),
                "tokens": token_rows,
                "joint_log_probability": float(candidate.drafter_log_score),
                "joint_probability": raw_joint,
                "token_probability_product": product,
                "normalized_probability_over_returned_blocks": float(normalized[index]),
            }
        )
    return {
        "tree_depth": 1,
        "prompt": prompt,
        "prompt_token_count": int(prompt_token_count),
        "block_size": int(block_size),
        "num_block_candidates": len(blocks),
        "per_position_topk": int(per_position_topk),
        "retained_mass_per_position": list(draft.retained_mass_per_position),
        "candidate_mass": float(draft.candidate_mass),
        "probability_semantics": (
            "Fast-dLLM single fully-masked-block marginal surrogate: "
            "q(B|C) ~= product_i q_i(x_i|C)."
        ),
        "blocks": blocks,
    }


def format_first_step_tree(report: dict[str, Any]) -> str:
    lines = [
        "=" * 100,
        "FAST-dLLM FIRST-STEP DRAFT TREE (no verifier)",
        "=" * 100,
        f"prompt_tokens={report['prompt_token_count']} block_size={report['block_size']} "
        f"candidates={report['num_block_candidates']} per_position_topk={report['per_position_topk']}",
        "retained_mass_per_position=["
        + ", ".join(f"{mass:.8f}" for mass in report["retained_mass_per_position"])
        + "]",
        f"sum_raw_joint_probability_of_returned_blocks={report['candidate_mass']:.12e}",
        "",
        "ROOT: verified prompt prefix",
    ]
    for block in report["blocks"]:
        branch = "└──" if block["rank"] == report["num_block_candidates"] else "├──"
        lines.append(
            f"{branch} BLOCK #{block['rank']} ids={block['token_ids']} text={block['decoded_block']!r}"
        )
        lines.append(
            f"    joint_q={block['joint_probability']:.12e}  "
            f"log_joint_q={block['joint_log_probability']:.8f}  "
            f"q_normalized_top{report['num_block_candidates']}="
            f"{block['normalized_probability_over_returned_blocks']:.8f}"
        )
        for token in block["tokens"]:
            lines.append(
                f"    token[{token['position']}] id={token['token_id']} "
                f"piece={token['token_piece']!r} decoded={token['decoded']!r} "
                f"p={token['probability']:.12e} logp={token['log_probability']:.8f}"
            )
        lines.append(
            "    check_product="
            f"{block['token_probability_product']:.12e} "
            "(must equal joint_q up to floating-point error)"
        )
    lines.extend(
        [
            "",
            "NOTE: joint_q is a surrogate product of masked-position marginals,",
            "not the exact probability of Fast-dLLM's iterative diffusion trajectory.",
            "=" * 100,
        ]
    )
    return "\n".join(lines)


def save_first_step_tree_report(report: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_detail(candidate, tokenizer, normalized_probability: float, rank: int) -> dict[str, Any]:
    tokens = []
    for position, (token_id, log_probability) in enumerate(
        zip(candidate.token_ids, candidate.drafter_token_logprobs), start=1
    ):
        token_piece, decoded = _decode_token(tokenizer, token_id)
        tokens.append(
            {
                "position": position,
                "token_id": int(token_id),
                "token_piece": token_piece,
                "decoded": decoded,
                "log_probability": float(log_probability),
                "probability": math.exp(float(log_probability)),
            }
        )
    return {
        "rank_at_parent": rank,
        "token_ids": list(candidate.token_ids),
        "decoded_block": tokenizer.decode(list(candidate.token_ids), skip_special_tokens=False).replace("\n", "\\n"),
        "tokens": tokens,
        "joint_log_probability": float(candidate.drafter_log_score),
        "joint_probability": math.exp(float(candidate.drafter_log_score)),
        "token_probability_product": math.prod(token["probability"] for token in tokens),
        "normalized_probability_at_parent": float(normalized_probability),
    }


def build_multi_step_tree_report(
    *,
    prompt: str,
    prompt_token_count: int,
    tree,
    tokenizer,
    block_size: int,
    per_position_topk: int,
    width_schedule: list[int],
) -> dict[str, Any]:
    """Serialize an already-built asymmetric drafter tree and all local supports."""
    nodes = []
    candidate_sets = []
    for node_id in sorted(tree.nodes):
        node = tree.nodes[node_id]
        if node.candidate_set:
            local_logs = torch.tensor(
                [candidate.drafter_log_score for candidate in node.candidate_set], dtype=torch.float64
            )
            local_normalized = normalize_log_scores(local_logs)
            child_by_block = {
                tree.nodes[child_id].block_token_ids: child_id for child_id in node.children
            }
            local_candidates = []
            for index, candidate in enumerate(node.candidate_set):
                detail = _candidate_detail(candidate, tokenizer, float(local_normalized[index]), index + 1)
                detail["expanded"] = candidate.token_ids in child_by_block
                detail["child_node_id"] = child_by_block.get(candidate.token_ids)
                local_candidates.append(detail)
            candidate_sets.append(
                {
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "candidate_mass": sum(math.exp(candidate.drafter_log_score) for candidate in node.candidate_set),
                    "candidates": local_candidates,
                }
            )
        if node.parent_id is None:
            nodes.append(
                {
                    "node_id": node.node_id,
                    "parent_id": None,
                    "depth": 0,
                    "children": list(node.children),
                    "block": None,
                    "cumulative_log_probability": 0.0,
                }
            )
            continue
        parent = tree.nodes[node.parent_id]
        parent_logs = torch.tensor(
            [candidate.drafter_log_score for candidate in parent.candidate_set], dtype=torch.float64
        )
        parent_normalized = normalize_log_scores(parent_logs)
        matches = [
            (index, candidate)
            for index, candidate in enumerate(parent.candidate_set)
            if candidate.token_ids == node.block_token_ids
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely map node {node.node_id} to its parent candidate")
        candidate_index, candidate = matches[0]
        nodes.append(
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "children": list(node.children),
                "block": _candidate_detail(
                    candidate, tokenizer, float(parent_normalized[candidate_index]), candidate_index + 1
                ),
                "cumulative_log_probability": float(node.cumulative_drafter_log_score),
                "cumulative_probability": math.exp(float(node.cumulative_drafter_log_score)),
            }
        )
    by_id = {node["node_id"]: node for node in nodes}
    deepest_paths = []
    target_depth = len(width_schedule)
    for leaf in nodes:
        if leaf["depth"] != target_depth:
            continue
        path = []
        cursor = leaf
        while cursor["parent_id"] is not None:
            path.append(cursor["block"])
            cursor = by_id[cursor["parent_id"]]
        path.reverse()
        deepest_paths.append(
            {
                "leaf_node_id": leaf["node_id"],
                "blocks": path,
                "all_token_ids": [token_id for block in path for token_id in block["token_ids"]],
                "decoded_text": "".join(block["decoded_block"] for block in path),
                "cumulative_log_probability": leaf["cumulative_log_probability"],
                "cumulative_probability": leaf["cumulative_probability"],
            }
        )
    return {
        "tree_depth": target_depth,
        "prompt": prompt,
        "prompt_token_count": int(prompt_token_count),
        "block_size": int(block_size),
        "per_position_topk": int(per_position_topk),
        "width_schedule": list(width_schedule),
        "actual_widths": tree.widths(),
        "tree_nodes_including_root": len(nodes),
        "probability_semantics": (
            "Fast-dLLM single fully-masked-block marginal surrogate: "
            "q(B|C) ~= product_i q_i(x_i|C)."
        ),
        "nodes": nodes,
        "candidate_sets": candidate_sets,
        "deepest_paths": deepest_paths,
    }


def format_multi_step_tree(report: dict[str, Any]) -> str:
    nodes = {node["node_id"]: node for node in report["nodes"]}
    lines = [
        "=" * 110,
        "FAST-dLLM MULTI-STEP ASYMMETRIC DRAFT TREE (no verifier)",
        "=" * 110,
        f"prompt_tokens={report['prompt_token_count']} block_size={report['block_size']} "
        f"tree_depth={report['tree_depth']} width_schedule={report['width_schedule']} "
        f"actual_widths={report['actual_widths']} nodes={report['tree_nodes_including_root']}",
        "",
        "ROOT: verified prompt prefix",
    ]

    def render_children(parent_id: int, prefix: str) -> None:
        children = nodes[parent_id]["children"]
        for position, child_id in enumerate(children):
            child = nodes[child_id]
            block = child["block"]
            last = position == len(children) - 1
            branch = "└──" if last else "├──"
            lines.append(
                f"{prefix}{branch} depth={child['depth']} node={child_id} "
                f"block={block['decoded_block']!r} ids={block['token_ids']} "
                f"joint_q={block['joint_probability']:.8e} "
                f"q_local={block['normalized_probability_at_parent']:.6f} "
                f"cum_log_q={child['cumulative_log_probability']:.6f}"
            )
            token_prefix = prefix + ("    " if last else "│   ")
            for token in block["tokens"]:
                lines.append(
                    f"{token_prefix}token[{token['position']}] id={token['token_id']} "
                    f"text={token['decoded']!r} p={token['probability']:.8e} "
                    f"logp={token['log_probability']:.6f}"
                )
            render_children(child_id, token_prefix)

    render_children(0, "")
    lines.extend(["", f"FULL {report['tree_depth']}-BLOCK PATHS RETAINED AT FINAL DEPTH"])
    for index, path in enumerate(report["deepest_paths"], start=1):
        lines.append(
            f"PATH #{index} leaf_node={path['leaf_node_id']} text={path['decoded_text']!r} "
            f"cumulative_q={path['cumulative_probability']:.8e} "
            f"cumulative_log_q={path['cumulative_log_probability']:.6f}"
        )
        for block_index, block in enumerate(path["blocks"], start=1):
            lines.append(
                f"  block[{block_index}] text={block['decoded_block']!r} ids={block['token_ids']} "
                f"joint_q={block['joint_probability']:.8e}"
            )
    lines.extend(
        [
            "",
            "JSON candidate_sets contains all five local candidates at every expanded node,",
            "including candidates pruned from the displayed expansion tree.",
            "=" * 110,
        ]
    )
    return "\n".join(lines)
