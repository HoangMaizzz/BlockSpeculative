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


class DrafterAdapter(ABC):
    tokenizer: object

    @abstractmethod
    def propose(self, prefix_ids: torch.LongTensor) -> DraftCandidates:
        raise NotImplementedError

