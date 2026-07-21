from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block_spec.decoder import BlockSpeculativeDecoder
from block_spec.drafter_tree_inspection import build_multi_step_tree_report
from block_spec.fast_dllm_adapter import FastDLLMv2Adapter
from block_spec.tokenizer_compatibility import validate_tokenizer_compatibility
from block_spec.tree_ar_scorer import ARTreeScorer
from block_spec.tree_builder import AsymmetricTreeBuilder
from block_spec.verifier_loader import load_local_causal_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build one Fast-dLLM tree, then traverse it once with an AR verifier"
    )
    parser.add_argument("--drafter-model-path", default=os.getenv("DRAFTER_MODEL_PATH"))
    parser.add_argument("--verifier-model-path", default=os.getenv("VERIFIER_MODEL_PATH"))
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--system-prompt", default="You are a careful mathematics tutor.")
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--num-block-candidates", type=int, default=5)
    parser.add_argument("--per-position-topk", type=int, default=8)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--width-schedule", default="5,20,10,5,1")
    parser.add_argument("--parent-cap-schedule", default="5,5,3,2,1")
    parser.add_argument("--depth2-min-children-per-parent", type=int, default=2)
    parser.add_argument("--max-tree-nodes", type=int, default=64)
    parser.add_argument("--max-tree-tokens", type=int, default=192)
    parser.add_argument("--native-block-size", type=int, default=32)
    parser.add_argument("--mask-token-id", type=int, default=151665)
    parser.add_argument("--proposal-mode", choices=("sample_q", "top1_q"), default="sample_q")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drafter-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--verifier-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--logsumexp-row-chunk-size", type=int, default=32)
    parser.add_argument("--output", default="outputs/prebuilt_tree_verification.json")
    parser.add_argument("--tree-output", default="outputs/prebuilt_drafter_tree.json")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def read_prompt(args) -> str:
    if bool(args.prompt) == bool(args.prompt_file):
        raise ValueError("Provide exactly one of --prompt or --prompt-file")
    text = args.prompt
    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
    text = (text or "").strip()
    if not text:
        raise ValueError("Prompt is empty")
    return text


def save_json(data, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    prompt = read_prompt(args)
    if not args.drafter_model_path or not args.verifier_model_path:
        raise ValueError("Set both DRAFTER_MODEL_PATH and VERIFIER_MODEL_PATH")
    width_schedule = [int(value) for value in args.width_schedule.split(",")]
    parent_caps = [int(value) for value in args.parent_cap_schedule.split(",")]
    if len(width_schedule) != args.depth:
        raise ValueError("width schedule length must equal --depth")

    print("\n[1/5] Loading Fast-dLLM drafter only...", flush=True)
    drafter_model, drafter_tokenizer = load_local_causal_model(
        args.drafter_model_path,
        dtype=args.drafter_dtype,
        device_map=args.device_map,
        local_files_only=True,
        allow_model_download=args.allow_model_download,
    )
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt_text = drafter_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = drafter_tokenizer(prompt_text, return_tensors="pt")["input_ids"].reshape(-1)
    print(f"Prompt received: {prompt!r}", flush=True)
    print(f"Prompt token count: {prefix_ids.numel()}", flush=True)
    adapter = FastDLLMv2Adapter(
        drafter_model,
        drafter_tokenizer,
        block_size=args.block_size,
        per_position_topk=args.per_position_topk,
        num_block_candidates=args.num_block_candidates,
        mask_token_id=args.mask_token_id,
        native_block_size=args.native_block_size,
    )
    builder = AsymmetricTreeBuilder(
        adapter,
        width_schedule=width_schedule,
        parent_cap_schedule=parent_caps,
        depth2_min_children_per_parent=args.depth2_min_children_per_parent,
        max_tree_nodes=args.max_tree_nodes,
        max_tree_tokens=args.max_tree_tokens,
        confidence_mode="cumulative_log_score",
    )
    print("\n[2/5] Building the complete drafter tree before verification...", flush=True)
    tree = builder.build(prefix_ids)
    print(f"Tree complete: nodes={len(tree.nodes)} widths={tree.widths()}", flush=True)
    tree_report = build_multi_step_tree_report(
        prompt=prompt,
        prompt_token_count=prefix_ids.numel(),
        tree=tree,
        tokenizer=drafter_tokenizer,
        block_size=args.block_size,
        per_position_topk=args.per_position_topk,
        width_schedule=width_schedule,
    )
    save_json(tree_report, args.tree_output)

    # Phase 1 is complete. Release the drafter before loading Qwen so the one-pass
    # verifier stays on GPU instead of silently offloading layers to CPU.
    del builder, adapter, drafter_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"Drafter released; GPU allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB",
            flush=True,
        )

    print("\n[3/5] Loading Qwen verifier after the tree is complete...", flush=True)
    verifier_model, verifier_tokenizer = load_local_causal_model(
        args.verifier_model_path,
        dtype=args.verifier_dtype,
        device_map=args.device_map,
        local_files_only=True,
        allow_model_download=args.allow_model_download,
        attn_implementation="sdpa",
    )
    validate_tokenizer_compatibility(
        drafter_tokenizer,
        verifier_tokenizer,
        require_compatibility=True,
        report_path="outputs/tokenizer_compatibility.json",
    )
    config = {
        "seed": args.seed,
        "sampling": {
            "proposal_mode": args.proposal_mode,
            "temperature": 1.0,
            "top_k": None,
            "top_p": 1.0,
            "enable_bonus_token": False,
        },
        "residual": {
            "enabled": True,
            "epsilon": 1e-12,
            "mass_warning_threshold": 0.95,
            "strict_mass": False,
        },
        "logging": {
            "print_verification_trace": True,
            "print_candidate_table": True,
        },
    }
    decoder = BlockSpeculativeDecoder(
        None,
        None,
        verifier_tokenizer,
        None,
        config,
    )
    print("\n[4/5] Scoring every tree candidate with ONE ancestor-masked AR forward...", flush=True)
    tree_score = ARTreeScorer(
        verifier_model, logsumexp_row_chunk_size=args.logsumexp_row_chunk_size
    ).score_tree(prefix_ids, tree)
    print(
        f"Tree AR scoring complete: forward_calls={tree_score['verifier_forward_calls']} "
        f"candidate_blocks={tree_score['tree_candidate_blocks_scored']} "
        f"flattened_length={tree_score['flattened_sequence_length']} "
        f"mask_shape={tree_score['attention_mask_shape']} "
        f"restricted_lm_head={tree_score['restricted_lm_head']} "
        f"projected_logits={tree_score['projected_logit_count']} "
        f"full_vocab_logits={tree_score['full_vocabulary_logits_materialized']}",
        flush=True,
    )
    print("\n[5/5] Sampling over cached p/q values; model calls during traversal = 0...", flush=True)
    result = decoder.traverse_precomputed_tree(
        tree, prefix_ids, max_blocks=args.depth,
    )
    result["prompt"] = prompt
    result["proposal_mode"] = args.proposal_mode
    result["seed"] = args.seed
    result["tree_ar_scoring"] = tree_score
    save_json(result, args.output)
    print(f"\nCommitted text: {result['committed_text']!r}", flush=True)
    print(f"Saved verification report: {Path(args.output).resolve()}", flush=True)
    print(f"Saved exact prebuilt tree: {Path(args.tree_output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
