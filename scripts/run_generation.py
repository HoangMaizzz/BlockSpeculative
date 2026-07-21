from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from block_spec.ar_block_scorer import ARBlockScorer
from block_spec.config import GenerationConfig, load_config
from block_spec.decoder import BlockSpeculativeDecoder
from block_spec.fast_dllm_adapter import FastDLLMv2Adapter
from block_spec.metrics import summarize_generation
from block_spec.tokenizer_compatibility import validate_tokenizer_compatibility
from block_spec.tree_builder import AsymmetricTreeBuilder
from block_spec.verifier_loader import load_local_causal_model


def parser():
    p = argparse.ArgumentParser(description="Fast-dLLM v2 / Qwen block speculative decoding")
    p.add_argument("--config", default="configs/colab_balanced.yaml")
    p.add_argument("--drafter-model-path", default=os.getenv("DRAFTER_MODEL_PATH"))
    p.add_argument("--verifier-model-path", default=os.getenv("VERIFIER_MODEL_PATH"))
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--allow-model-download", action="store_true")
    p.add_argument("--disable-bonus-token", action="store_true")
    p.add_argument("--block-size", type=int)
    p.add_argument("--candidate-count", type=int)
    p.add_argument("--width-schedule", help="Comma-separated global widths, e.g. 5,20,10,5,2,1")
    p.add_argument("--proposal-mode", choices=("sample_q", "top1_q"))
    p.add_argument("--output", default=None)
    p.add_argument("--metrics-output", default=None)
    p.add_argument("--disable-verification-trace", action="store_true")
    return p


def build_decoder(args):
    cfg = load_config(args.config)
    dpath = args.drafter_model_path or cfg["drafter"].get("model_path")
    vpath = args.verifier_model_path or cfg["verifier"].get("model_path")
    if not dpath or not vpath:
        raise ValueError("Set DRAFTER_MODEL_PATH and VERIFIER_MODEL_PATH or pass both CLI paths")
    if args.disable_bonus_token:
        cfg["sampling"]["enable_bonus_token"] = False
    if args.disable_verification_trace:
        cfg.setdefault("logging", {})["print_verification_trace"] = False
    if args.block_size:
        cfg["drafter"]["block_size"] = args.block_size
    if args.candidate_count:
        cfg["drafter"]["num_block_candidates"] = args.candidate_count
    if args.width_schedule:
        cfg["tree"]["width_schedule"] = [int(x) for x in args.width_schedule.split(",")]
    if args.proposal_mode:
        cfg["sampling"]["proposal_mode"] = args.proposal_mode
    drafter_model, drafter_tok = load_local_causal_model(
        dpath, dtype=cfg["drafter"]["dtype"], device_map="auto",
        local_files_only=True, allow_model_download=args.allow_model_download,
    )
    verifier_model, verifier_tok = load_local_causal_model(
        vpath, dtype=cfg["verifier"]["dtype"], device_map=cfg["verifier"]["device_map"],
        local_files_only=cfg["verifier"].get("local_files_only", True),
        allow_model_download=args.allow_model_download,
    )
    output_dir = Path(cfg["logging"]["output_dir"])
    report = validate_tokenizer_compatibility(
        drafter_tok, verifier_tok,
        require_compatibility=cfg["tokenizers"].get("require_compatibility", True),
        report_path=output_dir / "tokenizer_compatibility.json",
    )
    d = cfg["drafter"]
    adapter = FastDLLMv2Adapter(
        drafter_model, drafter_tok, d["block_size"], d["per_position_topk"],
        d["num_block_candidates"], d.get("mask_token_id"), d.get("native_block_size", 32),
        d.get("joint_search_backend", "heap"), d.get("candidate_set_mode", "drafter_topk"),
    )
    t = cfg["tree"]
    builder = AsymmetricTreeBuilder(
        adapter, t["width_schedule"], t["parent_cap_schedule"],
        t["depth2_min_children_per_parent"], t["max_tree_nodes"],
        t["max_tree_tokens"], t["confidence_mode"],
    )
    scorer = ARBlockScorer(verifier_model, cfg["verifier"]["candidate_batch_size"])
    print(json.dumps({
        "drafter_model_class": drafter_model.__class__.__name__,
        "verifier_model_class": verifier_model.__class__.__name__,
        "drafter_tokenizer_class": drafter_tok.__class__.__name__,
        "verifier_tokenizer_class": verifier_tok.__class__.__name__,
        "tokenizer_compatible": report.compatible,
        "block_size": d["block_size"], "candidate_count": d["num_block_candidates"],
        "width_schedule": t["width_schedule"], "tree_node_budget": t["max_tree_nodes"],
        "verifier_dtype": cfg["verifier"]["dtype"],
        "gpu_device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "available_gpu_memory": (torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else None),
        "bonus_token": cfg["sampling"]["enable_bonus_token"], "residual_sampling": cfg["residual"]["enabled"],
    }, indent=2))
    return BlockSpeculativeDecoder(adapter, scorer, verifier_tok, builder, cfg), cfg


def main():
    args = parser().parse_args()
    decoder, cfg = build_decoder(args)
    prompt = args.prompt
    if hasattr(decoder.tokenizer, "apply_chat_template"):
        prompt = decoder.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    max_tokens = args.max_new_tokens or cfg["generation"]["max_new_tokens"]
    result = decoder.generate(prompt, GenerationConfig(max_new_tokens=max_tokens, seed=cfg["seed"]))
    print(result.text)
    output = args.output or str(Path(cfg["logging"]["output_dir"]) / "generation.jsonl")
    decoder.save_jsonl(result, output)
    metrics = summarize_generation(result)
    print(json.dumps(metrics, indent=2))
    if args.metrics_output:
        target = Path(args.metrics_output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
