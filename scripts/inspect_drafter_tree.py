from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block_spec.drafter_tree_inspection import (
    build_first_step_tree_report,
    build_multi_step_tree_report,
    format_first_step_tree,
    format_multi_step_tree,
    save_first_step_tree_report,
)
from block_spec.fast_dllm_adapter import FastDLLMv2Adapter
from block_spec.tree_builder import AsymmetricTreeBuilder
from block_spec.verifier_loader import load_local_causal_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print a one-level or multi-step Fast-dLLM draft tree without loading a verifier"
    )
    parser.add_argument("--drafter-model-path", default=os.getenv("DRAFTER_MODEL_PATH"))
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--system-prompt", default="You are a careful mathematics tutor.")
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--num-block-candidates", type=int, default=5)
    parser.add_argument("--per-position-topk", type=int, default=8)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument(
        "--width-schedule",
        default=None,
        help="Comma-separated active path widths. For depth 5 the default is 5,20,10,5,1.",
    )
    parser.add_argument("--parent-cap-schedule", default="5,5,3,2,1")
    parser.add_argument("--depth2-min-children-per-parent", type=int, default=2)
    parser.add_argument("--max-tree-nodes", type=int, default=64)
    parser.add_argument("--max-tree-tokens", type=int, default=192)
    parser.add_argument("--native-block-size", type=int, default=32)
    parser.add_argument("--mask-token-id", type=int, default=151665)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output", default="outputs/drafter_tree_step1.json")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def read_prompt(args) -> str:
    if bool(args.prompt) == bool(args.prompt_file):
        raise ValueError("Provide exactly one of --prompt or --prompt-file")
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt is empty")
    return prompt


def main():
    args = parse_args()
    prompt = read_prompt(args)
    if not args.drafter_model_path:
        raise ValueError("Set DRAFTER_MODEL_PATH or pass --drafter-model-path")
    model, tokenizer = load_local_causal_model(
        args.drafter_model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=True,
        allow_model_download=args.allow_model_download,
    )
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].reshape(-1)
    adapter = FastDLLMv2Adapter(
        model=model,
        tokenizer=tokenizer,
        block_size=args.block_size,
        per_position_topk=args.per_position_topk,
        num_block_candidates=args.num_block_candidates,
        mask_token_id=args.mask_token_id,
        native_block_size=args.native_block_size,
    )
    if args.depth < 1:
        raise ValueError("--depth must be at least 1")
    if args.depth == 1:
        draft = adapter.propose(prefix_ids)
        report = build_first_step_tree_report(
            prompt=prompt,
            prompt_token_count=prefix_ids.numel(),
            draft=draft,
            tokenizer=tokenizer,
            block_size=args.block_size,
            per_position_topk=args.per_position_topk,
        )
        rendered = format_first_step_tree(report)
    else:
        if args.width_schedule:
            width_schedule = [int(value) for value in args.width_schedule.split(",")]
        elif args.depth == 5:
            width_schedule = [5, 20, 10, 5, 1]
        else:
            width_schedule = [args.num_block_candidates] * args.depth
        if len(width_schedule) != args.depth:
            raise ValueError("width schedule length must equal --depth")
        parent_caps = [int(value) for value in args.parent_cap_schedule.split(",")]
        builder = AsymmetricTreeBuilder(
            adapter,
            width_schedule=width_schedule,
            parent_cap_schedule=parent_caps,
            depth2_min_children_per_parent=args.depth2_min_children_per_parent,
            max_tree_nodes=args.max_tree_nodes,
            max_tree_tokens=args.max_tree_tokens,
            confidence_mode="cumulative_log_score",
        )
        tree = builder.build(prefix_ids)
        report = build_multi_step_tree_report(
            prompt=prompt,
            prompt_token_count=prefix_ids.numel(),
            tree=tree,
            tokenizer=tokenizer,
            block_size=args.block_size,
            per_position_topk=args.per_position_topk,
            width_schedule=width_schedule,
        )
        rendered = format_multi_step_tree(report)
    print(rendered, flush=True)
    save_first_step_tree_report(report, args.output)
    print(f"Saved JSON report: {Path(args.output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
