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


def decide_self_selection_acceptance(
    proposed_index: int,
    proposed_block: tuple[int, ...],
    p_probs: torch.Tensor,
    q_probs: torch.Tensor,
    generator: torch.Generator,
) -> AcceptanceDecision:
    """Accept a proposed block with its normalized verifier probability.

    Together with verifier sampling conditioned on rejecting this exact block,
    this preserves ``p_probs`` on the retained union support.  ``q_probs`` is
    recorded for diagnostics only; it does not enter the acceptance ratio.
    """
    probability = float(p_probs[proposed_index].clamp(0.0, 1.0))
    q_probability = float(q_probs[proposed_index].clamp_min(0.0))
    log_p = math.log(probability) if probability > 0.0 else -math.inf
    log_q = math.log(q_probability) if q_probability > 0.0 else -math.inf
    uniform = float(torch.rand((), generator=generator))
    return AcceptanceDecision(
        proposed_index,
        proposed_block,
        log_p,
        log_q,
        log_p,
        probability,
        uniform,
        uniform <= probability,
    )
