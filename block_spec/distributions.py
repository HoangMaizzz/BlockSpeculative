from __future__ import annotations

import torch


def normalize_log_scores(log_scores: torch.Tensor, tolerance: float = 1e-5) -> torch.Tensor:
    # Float64 avoids loss of the fractional log-normalizer when every score has
    # a very large magnitude (for example, around -10000 in unit tests).
    scores = log_scores.double()
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("log_scores must be a non-empty vector")
    if not torch.isfinite(scores).all():
        raise ValueError("log_scores contain NaN or infinity")
    probabilities = torch.exp(scores - torch.logsumexp(scores, dim=0))
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("normalization produced NaN or infinity")
    if abs(probabilities.sum().item() - 1.0) >= tolerance:
        raise RuntimeError("normalized probabilities do not sum to one")
    return probabilities


def compute_residual_distribution(
    p_probs: torch.Tensor, q_probs: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, bool]:
    if p_probs.shape != q_probs.shape or p_probs.ndim != 1:
        raise ValueError("p_probs and q_probs must be vectors with equal shape")
    residual = torch.clamp(p_probs.float() - q_probs.float(), min=0.0)
    total = residual.sum()
    if torch.isfinite(total) and total.item() > eps:
        return residual / total, False
    fallback = p_probs.float().clamp_min(0)
    fallback_total = fallback.sum()
    if not torch.isfinite(fallback_total) or fallback_total.item() <= eps:
        fallback = torch.ones_like(fallback) / fallback.numel()
    else:
        fallback = fallback / fallback_total
    return fallback, True
