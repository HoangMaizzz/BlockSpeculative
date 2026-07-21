# Repository inspection

## Initial state

The supplied directory was not a Git working tree and contained one file: `load_model.ipynb`. There was no local Fast-dLLM source, requirements file, CLI, test, cache wrapper, or speculative decoder to preserve. The notebook is therefore the local source of truth; the official model repository was used only to confirm its remote-code API.

## Exact models and APIs

The notebook loads `Efficient-Large-Model/Fast_dLLM_v2_1.5B` through `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and calls:

```python
model_drafter.generate(
    input_ids, tokenizer=tokenizer_drafter, max_new_tokens=...,
    small_block_size=8, threshold=0.9,
)
```

It is **Fast-dLLM v2**, a Qwen2.5-1.5B-based block-diffusion model, not v1, LLaDA, or Dream. The verifier is `Qwen/Qwen2.5-7B-Instruct`, exposed as a standard Transformers causal LM. Both notebook inputs have shape `[batch, sequence]`; both model logits have shape `[batch, sequence, vocabulary]`.

The official v2 `generate` signature includes `mask_id=151665`, `small_block_size=8`, `block_size=32`, `threshold`, `use_block_cache`, and `temperature`. It pads the unfinished native block with mask IDs, calls `forward(..., block_size=32)`, shifts logits one position, and progressively unmasks tokens. `forward` returns `CausalLMOutputWithPastAndBlockCache` with `logits`, `past_key_values`, and `block_past_key_values`.

## Adapter decision

`FastDLLMv2Adapter` uses the real remote-code model's `forward` method. It pads to the next native 32-token diffusion block with mask token 151665, performs the official one-position logit shift, and exposes the first fixed speculative block's marginal log probabilities with shape `[speculative_block_size, vocabulary]`. The heap search converts these to candidate blocks. This one-pass product is a surrogate and is not the probability of the model's iterative threshold/unmask trajectory.

The built-in hierarchical block/sub-block caching remains inside the original generation implementation. The adapter currently uses `use_cache=False` because speculative prefixes diverge across tree nodes; this is conservative and avoids mixing incompatible cache states. The AR scorer batches blocks sharing one prefix and dynamically chunks on OOM; it does not assume a cache representation across arbitrary remote-code model versions.

## Tokenizers

Both checkpoints derive from Qwen2.5, but compatibility is never assumed. Full token-string-to-ID maps, core special IDs, vocabulary sizes, and drafter-only mask tokens are checked. The report is saved under `outputs/tokenizer_compatibility.json`; generation stops in strict mode on any material mismatch.

Sources consulted: the official [model card](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B), [remote modeling code](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B/blob/main/modeling.py), and [NVlabs repository](https://github.com/NVlabs/Fast-dLLM).

