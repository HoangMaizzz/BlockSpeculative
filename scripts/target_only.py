from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from block_spec.verifier_loader import load_local_causal_model


def main():
    p = argparse.ArgumentParser(description="Qwen target-only baseline")
    p.add_argument("--verifier-model-path", default=os.getenv("VERIFIER_MODEL_PATH"), required=False)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--metrics-output")
    args = p.parse_args()
    if not args.verifier_model_path:
        p.error("set VERIFIER_MODEL_PATH or pass --verifier-model-path")
    model, tokenizer = load_local_causal_model(args.verifier_model_path, dtype=args.dtype)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False,
                                           add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.get_input_embeddings().weight.device)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - started
    count = output.shape[1] - ids.shape[1]
    print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True))
    metrics = {"mode": "target_only", "elapsed_seconds": elapsed, "generated_tokens": count,
               "tokens_per_second": count / max(elapsed, 1e-9),
               "peak_allocated_gpu_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
               "peak_reserved_gpu_memory": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
    print(json.dumps(metrics, indent=2))
    if args.metrics_output:
        target = Path(args.metrics_output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
