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

