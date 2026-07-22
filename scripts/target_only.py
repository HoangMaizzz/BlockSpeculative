from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from block_spec.prompting import PROMPT_STYLES, build_messages, render_chat_prompt
from block_spec.verifier_loader import load_local_causal_model


def main():
    p = argparse.ArgumentParser(description="Qwen target-only baseline")
    p.add_argument("--verifier-model-path", default=os.getenv("VERIFIER_MODEL_PATH"), required=False)
    p.add_argument("--prompt")
    p.add_argument("--prompt-file")
    p.add_argument("--system-prompt", default="You are a careful mathematics tutor.")
    p.add_argument("--prompt-style", choices=PROMPT_STYLES, default="system_user")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--metrics-output")
    args = p.parse_args()
    if not args.verifier_model_path:
        p.error("set VERIFIER_MODEL_PATH or pass --verifier-model-path")
    if bool(args.prompt) == bool(args.prompt_file):
        p.error("provide exactly one of --prompt or --prompt-file")
    user_prompt = args.prompt
    if args.prompt_file:
        user_prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    model, tokenizer = load_local_causal_model(
        args.verifier_model_path, dtype=args.dtype, attn_implementation="sdpa"
    )
    messages = build_messages(
        user_prompt,
        prompt_style=args.prompt_style,
        system_prompt=args.system_prompt,
    )
    prompt = render_chat_prompt(tokenizer, messages)
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.get_input_embeddings().weight.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    count = output.shape[1] - ids.shape[1]
    generated_text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    print(generated_text)
    metrics = {"mode": "target_only", "prompt_style": args.prompt_style,
               "prompt_messages": messages, "rendered_prompt": prompt,
               "elapsed_seconds": elapsed, "generated_tokens": count,
               "tokens_per_second": count / max(elapsed, 1e-9),
               "peak_allocated_gpu_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
               "peak_reserved_gpu_memory": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0}
    metrics["text"] = generated_text
    print(json.dumps(metrics, indent=2))
    if args.metrics_output:
        target = Path(args.metrics_output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
