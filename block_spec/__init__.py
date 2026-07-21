"""Block-level speculative decoding with Fast-dLLM v2 and an AR verifier."""

from .config import GenerationConfig
from .decoder import BlockSpeculativeDecoder
from .types import BlockCandidate, GenerationResult

__all__ = ["BlockSpeculativeDecoder", "BlockCandidate", "GenerationConfig", "GenerationResult"]
