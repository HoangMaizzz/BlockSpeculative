from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BlockCandidate:
    token_ids: tuple[int, ...]
    drafter_token_logprobs: tuple[float, ...]
    drafter_log_score: float
    verifier_token_logprobs: tuple[float, ...] | None = None
    verifier_log_score: float | None = None
    q_normalized: float | None = None
    p_normalized: float | None = None
    residual_probability: float | None = None
    confidence: float | None = None
    rank: int | None = None


@dataclass
class AcceptanceDecision:
    proposed_index: int
    proposed_block: tuple[int, ...]
    log_p: float
    log_q: float
    log_acceptance_probability: float
    acceptance_probability: float
    uniform_sample: float
    accepted: bool


@dataclass
class DraftTreeNode:
    node_id: int
    parent_id: int | None
    depth: int
    block_token_ids: tuple[int, ...] | None
    local_drafter_log_score: float
    cumulative_drafter_log_score: float
    confidence: float
    candidate_set: list[BlockCandidate] = field(default_factory=list)
    expanded_candidate_indices: list[int] = field(default_factory=list)
    children: list[int] = field(default_factory=list)
    is_leaf: bool = True


@dataclass
class NodeVerificationResult:
    node_id: int
    proposed_candidate_index: int
    accepted: bool
    committed_candidate_index: int
    committed_from_residual: bool
    continued_to_child: bool
    stop_reason: str | None


@dataclass
class ARBlockScoreResult:
    token_logprobs: Any
    block_logprobs: Any
    final_logits: Any | None = None
    batch_size_used: int | None = None


@dataclass
class TokenizerCompatibilityReport:
    compatible: bool
    reason: str
    drafter_vocab_size: int
    verifier_vocab_size: int
    mismatched_tokens: list[dict[str, Any]] = field(default_factory=list)
    drafter_special_ids: dict[str, int | None] = field(default_factory=dict)
    verifier_special_ids: dict[str, int | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_token_count: int
    generated_token_count: int
    elapsed_seconds: float
    rounds: list[dict[str, Any]]

