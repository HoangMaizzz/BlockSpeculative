from __future__ import annotations

import inspect

import torch

from .candidate_search import heap_topk_joint
from .drafter_adapter import DraftCandidates, DrafterAdapter


class FastDLLMv2Adapter(DrafterAdapter):
    """Adapter for the official Fast_dLLM_QwenForCausalLM remote-code model.

    A single fully-masked native block forward produces marginal logits.  Their
    product is a documented surrogate score, not the iterative trajectory
    probability used by ``model.generate``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        block_size: int = 5,
        per_position_topk: int = 8,
        num_block_candidates: int = 5,
        mask_token_id: int | None = None,
        native_block_size: int = 32,
        joint_search_backend: str = "heap",
        candidate_set_mode: str = "drafter_topk",
    ):
        if "Fast_dLLM" not in model.__class__.__name__:
            raise TypeError(f"Expected Fast-dLLM v2 model, got {model.__class__.__name__}")
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.per_position_topk = per_position_topk
        self.num_block_candidates = num_block_candidates
        self.mask_token_id = int(mask_token_id if mask_token_id is not None else 151665)
        self.native_block_size = native_block_size
        if joint_search_backend != "heap":
            raise ValueError("The first implementation supports joint_search_backend=heap")
        if candidate_set_mode != "drafter_topk":
            raise ValueError("Only tested candidate_set_mode=drafter_topk is enabled")
        if block_size > native_block_size:
            raise ValueError("speculative block_size cannot exceed Fast-dLLM native block size")
        signature = inspect.signature(model.forward)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if "block_size" not in signature.parameters and not accepts_kwargs:
            raise TypeError("Fast-dLLM forward does not expose the required block_size API")

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def marginal_log_probs(self, prefix_ids: torch.LongTensor) -> torch.Tensor:
        prefix = prefix_ids.reshape(1, -1).to(self.device)
        remaining = self.native_block_size - (prefix.shape[1] % self.native_block_size)
        if remaining < self.block_size:
            # Complete the current native block before taking the next fixed block.
            remaining += self.native_block_size
        masks = torch.full((1, remaining), self.mask_token_id, dtype=torch.long, device=self.device)
        model_input = torch.cat((prefix, masks), dim=1)
        output = self.model(input_ids=model_input, use_cache=False, block_size=self.native_block_size)
        logits = output.logits
        # This is the same token shift used by the official v2 generate method.
        shifted = torch.cat((logits[:, :1, :], logits[:, :-1, :]), dim=1)
        start = prefix.shape[1]
        selected = shifted[0, start : start + self.block_size].float()
        if selected.shape[0] != self.block_size:
            raise RuntimeError("Fast-dLLM did not return a complete masked block")
        return torch.log_softmax(selected, dim=-1)

    def propose(self, prefix_ids: torch.LongTensor) -> DraftCandidates:
        log_probs = self.marginal_log_probs(prefix_ids)
        result = heap_topk_joint(
            log_probs, self.per_position_topk, self.num_block_candidates
        )
        return DraftCandidates(
            result.candidates,
            result.retained_mass_per_position,
            result.candidate_mass,
            # FP16 keeps the complete lookup table small. Retain it beside the
            # drafter so verifier-only union blocks require no full-table host transfer.
            log_probs.to(dtype=torch.float16).contiguous(),
        )
