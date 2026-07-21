from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .types import BlockCandidate


@dataclass
class DraftCandidates:
    candidates: list[BlockCandidate]
    retained_mass_per_position: tuple[float, ...]
    candidate_mass: float
    # Full q_i(token) table retained on the drafter device so verifier-only
    # blocks can be scored later without another drafter call.
    # Shape: [block_size, vocab].
    marginal_log_probs: torch.Tensor | None = None


class DrafterAdapter(ABC):
    tokenizer: object

    @abstractmethod
    def propose(self, prefix_ids: torch.LongTensor) -> DraftCandidates:
        raise NotImplementedError
