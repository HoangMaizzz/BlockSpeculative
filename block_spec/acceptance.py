from __future__ import annotations

import math

import torch

from .types import AcceptanceDecision


def select_proposal(q_probs: torch.Tensor, mode: str, generator: torch.Generator) -> int:
    if mode == "top1_q":
        return int(torch.argmax(q_probs))
    if mode == "sample_q":
        return int(torch.multinomial(q_probs.cpu(), 1, generator=generator))
    raise ValueError(f"Unsupported proposal mode: {mode}")


def decide_acceptance(
    proposed_index: int,
    proposed_block: tuple[int, ...],
    p_probs: torch.Tensor,
    q_probs: torch.Tensor,
    generator: torch.Generator,
) -> AcceptanceDecision:
    log_p = float(torch.log(p_probs[proposed_index].clamp_min(torch.finfo(torch.float32).tiny)))
    log_q = float(torch.log(q_probs[proposed_index].clamp_min(torch.finfo(torch.float32).tiny)))
    log_alpha = min(0.0, log_p - log_q)
    uniform = float(torch.rand((), generator=generator).clamp_min(torch.finfo(torch.float32).tiny))
    return AcceptanceDecision(
        proposed_index, proposed_block, log_p, log_q, log_alpha,
        math.exp(log_alpha), uniform, math.log(uniform) <= log_alpha,
    )

