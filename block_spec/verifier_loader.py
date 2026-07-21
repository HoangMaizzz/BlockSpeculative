from __future__ import annotations

from pathlib import Path

import torch


def resolve_dtype(name: str):
    choices = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in choices:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(choices)}")
    return choices[name]


def load_local_causal_model(
    path: str,
    *,
    dtype: str = "float16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    allow_model_download: bool = False,
    attn_implementation: str | None = None,
    max_memory: dict | None = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = Path(path).expanduser().resolve() if Path(path).exists() else None
    if local_files_only and resolved is None and not allow_model_download:
        raise FileNotFoundError(
            f"Local model path does not exist: {path}. Supply a Colab directory or pass --allow-model-download."
        )
    source = str(resolved) if resolved is not None else path
    effective_local = local_files_only and resolved is not None
    print(f"Resolved model path: {source}")
    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=trust_remote_code, local_files_only=effective_local
    )
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolve_dtype(dtype),
        "device_map": device_map,
        "local_files_only": effective_local,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs).eval()
    print(f"Model class: {model.__class__.__name__}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")
    if torch.cuda.is_available():
        print(f"GPU allocated after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    return model, tokenizer
