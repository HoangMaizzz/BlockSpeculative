from __future__ import annotations

import statistics

import torch


def summarize_generation(result) -> dict[str, float | int]:
    rounds = result.rounds
    node_rows = [n for r in rounds for n in r.get("node_results", [])]
    total = max(result.elapsed_seconds, 1e-12)
    rejected = sum(not n.get("accepted", False) for n in node_rows)
    residual = [n for n in node_rows if n.get("residual_used")]
    return {
        "total_generation_time": result.elapsed_seconds,
        "tokens_per_second": result.generated_token_count / total,
        "time_to_first_token": ((rounds[0].get("draft_time_ms", 0) + rounds[0].get("verify_time_ms", 0)) / 1000
                                if rounds else 0.0),
        "drafter_latency_ms": sum(r.get("draft_time_ms", 0) for r in rounds),
        "verifier_latency_ms": sum(r.get("verify_time_ms", 0) for r in rounds),
        "tree_construction_latency_ms": sum(r.get("draft_time_ms", 0) for r in rounds),
        "residual_arithmetic_latency_ms": sum(n.get("residual_time_ms", 0) for n in node_rows),
        "average_committed_tokens_per_round": statistics.fmean(r.get("committed_tokens", 0) for r in rounds) if rounds else 0.0,
        "average_accepted_blocks_per_round": (sum(n.get("accepted", False) for n in node_rows) / len(rounds)) if rounds else 0.0,
        "rejection_rate": rejected / len(node_rows) if node_rows else 0.0,
        "residual_branch_continuation_rate": (sum(n.get("continued_in_tree", False) for n in residual) / len(residual)) if residual else 0.0,
        "residual_to_leaf_rate": (sum(not n.get("continued_in_tree", False) for n in residual) / len(residual)) if residual else 0.0,
        "bonus_token_frequency": sum(r.get("bonus_token_used", False) for r in rounds) / len(rounds) if rounds else 0.0,
        "mean_retained_candidate_mass_q": statistics.fmean(n["candidate_mass_q"] for n in node_rows) if node_rows else 0.0,
        "mean_retained_candidate_mass_p": statistics.fmean(n["candidate_mass_p"] for n in node_rows) if node_rows else 0.0,
        "peak_allocated_gpu_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "peak_reserved_gpu_memory": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
    }
