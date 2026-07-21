from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block_spec.decoder import BlockSpeculativeDecoder, sample_logits
from block_spec.fast_dllm_adapter import FastDLLMv2Adapter
from block_spec.tokenizer_compatibility import validate_tokenizer_compatibility
from block_spec.tree_ar_scorer import ARTreeScorer
from block_spec.tree_builder import AsymmetricTreeBuilder
from block_spec.verifier_candidate_search import (
    ARVerifierTopKBlockSearch,
    merge_verifier_topk_into_tree,
)
from block_spec.verifier_loader import load_local_causal_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full multi-round union-TopK block speculative generation"
    )
    parser.add_argument("--drafter-model-path", default=os.getenv("DRAFTER_MODEL_PATH"))
    parser.add_argument("--verifier-model-path", default=os.getenv("VERIFIER_MODEL_PATH"))
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--system-prompt", default="You are a careful mathematics tutor.")
    parser.add_argument("--max-new-tokens", type=int, default=126)
    parser.add_argument("--stop-on-final-answer", action="store_true")
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--num-block-candidates", type=int, default=5)
    parser.add_argument("--per-position-topk", type=int, default=8)
    parser.add_argument(
        "--candidate-set-mode",
        choices=("drafter_topk", "union_topk"),
        default="drafter_topk",
    )
    parser.add_argument("--verifier-block-topk", type=int, default=5)
    parser.add_argument("--verifier-beam-batch-size", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width-schedule", default="5,15,15")
    parser.add_argument("--parent-cap-schedule", default="5,5,5")
    parser.add_argument("--depth2-min-children-per-parent", type=int, default=0)
    parser.add_argument("--max-tree-nodes", type=int, default=48)
    parser.add_argument("--max-tree-tokens", type=int, default=128)
    parser.add_argument("--native-block-size", type=int, default=32)
    parser.add_argument("--mask-token-id", type=int, default=151665)
    parser.add_argument("--proposal-mode", choices=("sample_q", "top1_q"), default="sample_q")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drafter-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--verifier-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--verifier-max-gpu-memory-gib", type=int, default=14)
    parser.add_argument("--verifier-max-cpu-memory-gib", type=int, default=45)
    parser.add_argument("--output", default="outputs/full_union_generation.json")
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


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def verifier_one_token(model, prefix: torch.LongTensor, generator: torch.Generator) -> int:
    device = model.get_input_embeddings().weight.device
    ids = prefix.reshape(1, -1).to(device)
    base_model = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if base_model is not None and lm_head is not None:
        output = base_model(input_ids=ids, use_cache=False)
        logits = lm_head(output.last_hidden_state[:, -1])[0]
    else:
        output = model(input_ids=ids, use_cache=False)
        logits = output.logits[0, -1]
    return sample_logits(logits, 1.0, None, 1.0, generator)


def main():
    args = parse_args()
    prompt = read_prompt(args)
    if not args.drafter_model_path or not args.verifier_model_path:
        raise ValueError("Set DRAFTER_MODEL_PATH and VERIFIER_MODEL_PATH")
    widths = [int(value) for value in args.width_schedule.split(",")]
    parent_caps = [int(value) for value in args.parent_cap_schedule.split(",")]
    if len(widths) != args.depth:
        raise ValueError("width schedule length must equal --depth")

    print("[LOAD] Fast-dLLM drafter...", flush=True)
    drafter_model, drafter_tokenizer = load_local_causal_model(
        args.drafter_model_path,
        dtype=args.drafter_dtype,
        device_map=args.device_map,
        local_files_only=True,
        allow_model_download=args.allow_model_download,
    )
    print("[LOAD] Qwen verifier...", flush=True)
    verifier_model, verifier_tokenizer = load_local_causal_model(
        args.verifier_model_path,
        dtype=args.verifier_dtype,
        device_map=args.device_map,
        local_files_only=True,
        allow_model_download=args.allow_model_download,
        attn_implementation="sdpa",
        max_memory=(
            {
                0: f"{args.verifier_max_gpu_memory_gib}GiB",
                "cpu": f"{args.verifier_max_cpu_memory_gib}GiB",
            }
            if args.device_map == "auto" and torch.cuda.is_available()
            else None
        ),
    )
    validate_tokenizer_compatibility(
        drafter_tokenizer,
        verifier_tokenizer,
        require_compatibility=True,
        report_path="outputs/tokenizer_compatibility.json",
    )

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt_text = drafter_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix = drafter_tokenizer(prompt_text, return_tensors="pt")["input_ids"].reshape(-1).cpu()
    prompt_token_count = int(prefix.numel())

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
        width_schedule=widths,
        parent_cap_schedule=parent_caps,
        depth2_min_children_per_parent=args.depth2_min_children_per_parent,
        max_tree_nodes=args.max_tree_nodes,
        max_tree_tokens=args.max_tree_tokens,
        confidence_mode="cumulative_log_score",
    )
    verifier_search = None
    if args.candidate_set_mode == "union_topk":
        verifier_search = ARVerifierTopKBlockSearch(
            verifier_model,
            block_size=args.block_size,
            topk=args.verifier_block_topk,
            batch_size=args.verifier_beam_batch_size,
            vocab_limit=len(verifier_tokenizer),
        )
    tree_scorer = ARTreeScorer(verifier_model)
    decoder = BlockSpeculativeDecoder(
        None,
        None,
        verifier_tokenizer,
        None,
        {
            "seed": args.seed,
            "sampling": {"proposal_mode": args.proposal_mode},
            "residual": {"epsilon": 1e-12},
            "logging": {"print_verification_trace": True, "print_candidate_table": True},
        },
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    eos = verifier_tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    generated: list[int] = []
    round_reports: list[dict] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sync_cuda()
    started = time.perf_counter()
    stop_reason = "max_new_tokens"
    round_index = 0

    while len(generated) < args.max_new_tokens:
        sync_cuda()
        round_started = time.perf_counter()
        remaining = args.max_new_tokens - len(generated)
        print(
            f"\n{'#' * 96}\n[FULL ROUND {round_index}] "
            f"prefix_tokens={prefix.numel()} generated={len(generated)} remaining={remaining}",
            flush=True,
        )
        if remaining < args.block_size:
            token = verifier_one_token(verifier_model, prefix, generator)
            generated.append(token)
            prefix = torch.cat((prefix, torch.tensor([token], dtype=torch.long)))
            if token in eos_ids:
                stop_reason = "eos"
                break
            round_index += 1
            continue

        draft_started = time.perf_counter()
        tree = builder.build(prefix)
        sync_cuda()
        draft_seconds = time.perf_counter() - draft_started
        q_table_bytes = sum(
            node.drafter_marginal_logprobs.numel()
            * node.drafter_marginal_logprobs.element_size()
            for node in tree.nodes.values()
            if node.drafter_marginal_logprobs is not None
        )
        print(
            f"[TREE] nodes={len(tree.nodes)} widths={tree.widths()} "
            f"q_tables_device={adapter.device} q_tables={q_table_bytes / 2**20:.2f} MiB "
            f"draft_s={draft_seconds:.2f}",
            flush=True,
        )

        beams = None
        sync_cuda()
        beam_started = time.perf_counter()
        if verifier_search is not None:
            beams, beam_report = verifier_search.search_tree(prefix, tree)
            union_report = merge_verifier_topk_into_tree(tree, beams)
        else:
            beam_report = {
                "candidate_set_mode": "drafter_topk",
                "verifier_forward_calls": 0,
            }
            union_report = {
                "verifier_only_blocks_added": 0,
                "duplicate_blocks_merged": 0,
            }
        sync_cuda()
        beam_seconds = time.perf_counter() - beam_started
        score_started = time.perf_counter()
        score_report = tree_scorer.score_tree(prefix, tree)
        sync_cuda()
        tree_score_seconds = time.perf_counter() - score_started
        sampling_started = time.perf_counter()
        sampled = decoder.traverse_precomputed_tree(
            tree,
            prefix,
            max_blocks=min(args.depth, remaining // args.block_size),
        )
        sampling_seconds = time.perf_counter() - sampling_started
        committed = sampled["committed_token_ids"]
        if not committed:
            token = verifier_one_token(verifier_model, prefix, generator)
            committed = [token]
            sampled["stop_reason"] = "empty_tree_target_fallback"
        generated.extend(committed)
        prefix = torch.cat((prefix, torch.tensor(committed, dtype=torch.long)))
        sync_cuda()
        round_seconds = time.perf_counter() - round_started
        cumulative_seconds = time.perf_counter() - started
        round_tokens_per_second = len(committed) / max(round_seconds, 1e-9)
        cumulative_tokens_per_second = len(generated) / max(cumulative_seconds, 1e-9)
        print(
            f"[ROUND SUMMARY] round={round_index} "
            f"committed_blocks={sampled['blocks_committed']} "
            f"accepted_blocks={sampled['accepted_blocks']} "
            f"rejected_blocks={sampled['rejected_blocks']} "
            f"residual_blocks={sampled['residual_blocks']} "
            f"committed_tokens={len(committed)} "
            f"draft_s={draft_seconds:.3f} beam_s={beam_seconds:.3f} "
            f"tree_verify_s={tree_score_seconds:.3f} sampling_s={sampling_seconds:.6f} "
            f"round_s={round_seconds:.3f} round_tok_s={round_tokens_per_second:.3f} "
            f"cumulative_tok_s={cumulative_tokens_per_second:.3f} "
            f"stop_reason={sampled['stop_reason']}",
            flush=True,
        )
        round_reports.append(
            {
                "round": round_index,
                "tree_nodes": len(tree.nodes),
                "tree_widths": tree.widths(),
                "draft_seconds": draft_seconds,
                "verifier_beam_seconds": beam_seconds,
                "tree_verification_seconds": tree_score_seconds,
                "sampling_seconds": sampling_seconds,
                "round_seconds": round_seconds,
                "round_tokens_per_second": round_tokens_per_second,
                "cumulative_tokens_per_second": cumulative_tokens_per_second,
                "drafter_marginal_tables_device": str(adapter.device),
                "drafter_marginal_tables_bytes": q_table_bytes,
                "verifier_beam": beam_report,
                "union": union_report,
                "tree_scoring": score_report,
                "sampling": sampled,
            }
        )
        generated_text = verifier_tokenizer.decode(generated, skip_special_tokens=True)
        print(f"[FULL OUTPUT SO FAR]\n{generated_text}", flush=True)
        if any(token in eos_ids for token in committed):
            stop_reason = "eos"
            break
        if args.stop_on_final_answer and re.search(
            r"Final answer\s*:\s*[^\s]+", generated_text, flags=re.IGNORECASE
        ):
            stop_reason = "final_answer_pattern"
            break
        del tree
        if beams is not None:
            del beams
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        round_index += 1

    if any(token in eos_ids for token in generated):
        generated = generated[: next(i for i, token in enumerate(generated) if token in eos_ids) + 1]
    sync_cuda()
    elapsed_seconds = time.perf_counter() - started
    text = verifier_tokenizer.decode(generated, skip_special_tokens=True)
    total_committed_blocks = sum(
        report["sampling"]["blocks_committed"] for report in round_reports
    )
    total_accepted_blocks = sum(
        report["sampling"]["accepted_blocks"] for report in round_reports
    )
    total_rejected_blocks = sum(
        report["sampling"]["rejected_blocks"] for report in round_reports
    )
    total_residual_blocks = sum(
        report["sampling"]["residual_blocks"] for report in round_reports
    )
    result = {
        "prompt": prompt,
        "prompt_token_count": prompt_token_count,
        "generated_token_count": len(generated),
        "generated_token_ids": generated,
        "text": text,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": len(generated) / max(elapsed_seconds, 1e-9),
        "peak_allocated_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        "peak_reserved_gpu_memory_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        ),
        "total_committed_blocks": total_committed_blocks,
        "total_accepted_blocks": total_accepted_blocks,
        "total_rejected_blocks": total_rejected_blocks,
        "total_residual_blocks": total_residual_blocks,
        "rounds": round_reports,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'=' * 96}\nFINAL GENERATED TEXT:\n{text}", flush=True)
    print(
        f"[FINAL SPEED] tokens={len(generated)} elapsed_s={elapsed_seconds:.3f} "
        f"tokens_per_second={result['tokens_per_second']:.3f} "
        f"peak_allocated_gib={result['peak_allocated_gpu_memory_bytes'] / 2**30:.3f} "
        f"peak_reserved_gib={result['peak_reserved_gpu_memory_bytes'] / 2**30:.3f}",
        flush=True,
    )
    print(f"Saved: {target.resolve()}", flush=True)


if __name__ == "__main__":
    main()
