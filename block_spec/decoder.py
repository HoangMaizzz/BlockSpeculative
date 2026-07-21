from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import torch

from .acceptance import decide_acceptance, select_proposal
from .config import GenerationConfig
from .distributions import compute_residual_distribution, normalize_log_scores
from .types import GenerationResult, NodeVerificationResult


def sample_logits(logits, temperature, top_k, top_p, generator):
    scores = logits.float().cpu()
    if temperature <= 0:
        return int(scores.argmax())
    scores = scores / temperature
    if top_k:
        cutoff = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores[scores < cutoff] = -torch.inf
    if top_p < 1.0:
        sorted_scores, order = torch.sort(scores, descending=True)
        probs = torch.softmax(sorted_scores, -1)
        remove = torch.cumsum(probs, -1) - probs > top_p
        sorted_scores[remove] = -torch.inf
        scores = torch.full_like(scores, -torch.inf).scatter(0, order, sorted_scores)
    return int(torch.multinomial(torch.softmax(scores, -1), 1, generator=generator))


class BlockSpeculativeDecoder:
    def __init__(self, drafter, verifier_scorer, tokenizer, tree_builder, config: dict):
        self.drafter = drafter
        self.scorer = verifier_scorer
        self.tokenizer = tokenizer
        self.tree_builder = tree_builder
        self.config = config
        self.generator = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 42)))

    def _trace_enabled(self) -> bool:
        return bool(self.config.get("logging", {}).get("print_verification_trace", False))

    def _block_text(self, token_ids) -> str:
        try:
            text = self.tokenizer.decode(list(token_ids), skip_special_tokens=False)
            return text.replace("\n", "\\n")
        except Exception:
            return "<decode unavailable>"

    def _print_node_trace(
        self, node, decision, q_probs, p_probs, residual, committed, child,
        raw_q_mass, raw_p_mass, residual_fallback,
    ) -> None:
        if not self._trace_enabled():
            return
        print("\n" + "=" * 88)
        print(
            f"[VERIFY] node={node.node_id} depth={node.depth} candidates={len(node.candidate_set)} "
            f"mass_q={raw_q_mass:.6g} mass_p={raw_p_mass:.6g}"
        )
        if self.config.get("logging", {}).get("print_candidate_table", True):
            print(" idx  role       q_S         p_S         residual    log_q       log_p       block")
            for i, candidate in enumerate(node.candidate_set):
                roles = []
                if i == decision.proposed_index:
                    roles.append("PROPOSED")
                if i == committed:
                    roles.append("COMMIT")
                role = "+".join(roles) or "-"
                residual_value = float(residual[i]) if residual is not None else 0.0
                print(
                    f" {i:>3}  {role:<10} {float(q_probs[i]):>10.6f}  {float(p_probs[i]):>10.6f}  "
                    f"{residual_value:>10.6f}  {candidate.drafter_log_score:>10.4f}  "
                    f"{float(candidate.verifier_log_score):>10.4f}  {list(candidate.token_ids)} "
                    f"{self._block_text(candidate.token_ids)!r}"
                )
        verdict = "ACCEPT" if decision.accepted else "REJECT"
        print(
            f"[DECISION] proposed_index={decision.proposed_index} "
            f"alpha={decision.acceptance_probability:.6f} u={decision.uniform_sample:.6f} "
            f"log_alpha={decision.log_acceptance_probability:.6f} => {verdict}"
        )
        if not decision.accepted:
            print(
                f"[RESIDUAL] sampled_index={committed} fallback_to_p={str(residual_fallback).lower()} "
                f"probability={float(residual[committed]):.6f}"
            )
        print(
            f"[COMMIT] index={committed} tokens={list(node.candidate_set[committed].token_ids)} "
            f"text={self._block_text(node.candidate_set[committed].token_ids)!r} "
            f"next={'child node ' + str(child.node_id) if child is not None else 'stop at leaf'}"
        )

    def _target_token(self, prefix):
        device = self.scorer.device
        with torch.inference_mode():
            logits = self.scorer.model(input_ids=prefix.to(device), use_cache=False).logits[0, -1].cpu()
        s = self.config["sampling"]
        return sample_logits(logits, s.get("temperature", 1.0), s.get("top_k"), s.get("top_p", 1.0), self.generator)

    def _verify_node(self, tree, node, prefix):
        blocks = torch.tensor([c.token_ids for c in node.candidate_set], dtype=torch.long)
        scores = self.scorer.score_blocks(prefix, blocks)
        q_logs = torch.tensor([c.drafter_log_score for c in node.candidate_set])
        p_logs = scores.block_logprobs
        q_probs, p_probs = normalize_log_scores(q_logs), normalize_log_scores(p_logs)
        for i, candidate in enumerate(node.candidate_set):
            candidate.verifier_token_logprobs = tuple(float(x) for x in scores.token_logprobs[i])
            candidate.verifier_log_score = float(p_logs[i])
            candidate.q_normalized = float(q_probs[i])
            candidate.p_normalized = float(p_probs[i])
        mode = self.config["sampling"].get("proposal_mode", "sample_q")
        proposed = select_proposal(q_probs, mode, self.generator)
        decision = decide_acceptance(proposed, node.candidate_set[proposed].token_ids, p_probs, q_probs, self.generator)
        residual_used = False
        fallback = False
        residual = None
        committed = proposed
        if not decision.accepted:
            residual_started = time.perf_counter()
            residual, fallback = compute_residual_distribution(
                p_probs, q_probs, float(self.config["residual"].get("epsilon", 1e-12))
            )
            committed = int(torch.multinomial(residual, 1, generator=self.generator))
            for i, candidate in enumerate(node.candidate_set):
                candidate.residual_probability = float(residual[i])
            residual_used = True
            residual_time_ms = (time.perf_counter() - residual_started) * 1000
        else:
            residual_time_ms = 0.0
        child = tree.child_for_candidate(node, committed)
        result = NodeVerificationResult(
            node.node_id, proposed, decision.accepted, committed, residual_used,
            child is not None, None if child is not None else "committed_candidate_is_unexpanded_leaf",
        )
        raw_q_mass = sum(math.exp(x) for x in q_logs.tolist())
        raw_p_mass = sum(math.exp(x) for x in p_logs.tolist())
        self._print_node_trace(
            node, decision, q_probs, p_probs, residual, committed, child,
            raw_q_mass, raw_p_mass, fallback,
        )
        threshold = float(self.config["residual"].get("mass_warning_threshold", 0.95))
        if raw_q_mass < threshold or raw_p_mass < threshold:
            if self.config["residual"].get("strict_mass", False):
                raise RuntimeError(
                    f"Candidate mass below strict threshold: q={raw_q_mass:.4g}, p={raw_p_mass:.4g}, threshold={threshold}"
                )
            warnings.warn(
                f"Truncated support retains q={raw_q_mass:.4g}, p={raw_p_mass:.4g}; values are not full-space coverage proofs"
            )
        diagnostics = {
            "candidate_mass_q": raw_q_mass,
            "candidate_mass_p": raw_p_mass,
            "one_minus_drafter_mass": max(0.0, 1.0 - raw_q_mass),
            "one_minus_verifier_mass": max(0.0, 1.0 - raw_p_mass),
            "proposed_block": list(node.candidate_set[proposed].token_ids),
            "accepted": decision.accepted,
            "acceptance_probability": decision.acceptance_probability,
            "uniform_sample": decision.uniform_sample,
            "random_seed": int(self.config.get("seed", 42)),
            "residual_used": residual_used,
            "residual_sampled_candidate_index": committed if residual_used else None,
            "residual_degenerate_fallback": fallback,
            "residual_time_ms": residual_time_ms,
            "committed_block": list(node.candidate_set[committed].token_ids),
            "continued_in_tree": child is not None,
        }
        return result, diagnostics, child

    def verify_prebuilt_tree(
        self,
        tree,
        prefix_ids: torch.LongTensor,
        *,
        max_blocks: int | None = None,
        enable_bonus_token: bool = False,
    ) -> dict:
        """Verify exactly one already-built tree without invoking the drafter.

        Only candidate sets on the committed path are scored.  A residual-selected
        candidate continues when that candidate already owns a child in ``tree``;
        otherwise traversal stops at the corresponding pruned leaf.
        """
        prefix = prefix_ids.reshape(-1).long().cpu()
        original_prefix_length = int(prefix.numel())
        node = tree.root
        committed_tokens: list[int] = []
        decisions: list[dict] = []
        visited_node_ids: list[int] = []
        stop_reason = None
        blocks_committed = 0
        eos_ids = self.tokenizer.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        if self._trace_enabled():
            print(
                f"\n[PREBUILT TREE VERIFY] root={tree.root_id} nodes={len(tree.nodes)} "
                f"widths={tree.widths()} prefix_tokens={original_prefix_length}",
                flush=True,
            )
        while node.candidate_set:
            if max_blocks is not None and blocks_committed >= max_blocks:
                stop_reason = "max_blocks_reached"
                break
            visited_node_ids.append(node.node_id)
            result, diagnostics, child = self._verify_node(tree, node, prefix.reshape(1, -1))
            committed_candidate = node.candidate_set[result.committed_candidate_index]
            block = committed_candidate.token_ids
            committed_tokens.extend(block)
            prefix = torch.cat((prefix, torch.tensor(block, dtype=torch.long)))
            blocks_committed += 1
            decisions.append(
                {
                    **result.__dict__,
                    **diagnostics,
                    "depth": node.depth,
                    "committed_text": self._block_text(block),
                    "prefix_length_after_commit": int(prefix.numel()),
                }
            )
            if any(token_id in eos_ids for token_id in block):
                stop_reason = "eos_in_committed_block"
                break
            if child is None:
                stop_reason = "committed_candidate_has_no_expanded_child"
                break
            node = child
            if not node.candidate_set:
                stop_reason = "reached_final_expanded_depth"
                break
        if stop_reason is None:
            stop_reason = "tree_has_no_more_candidate_sets"
        bonus_token = None
        if (
            enable_bonus_token
            and stop_reason == "reached_final_expanded_depth"
        ):
            bonus_token = self._target_token(prefix.reshape(1, -1))
            committed_tokens.append(bonus_token)
            prefix = torch.cat((prefix, torch.tensor([bonus_token], dtype=torch.long)))
            if self._trace_enabled():
                print(
                    f"[BONUS] token={bonus_token} text={self._block_text((bonus_token,))!r}",
                    flush=True,
                )
        summary = {
            "tree_nodes": len(tree.nodes),
            "tree_widths": tree.widths(),
            "original_prefix_length": original_prefix_length,
            "visited_node_ids": visited_node_ids,
            "blocks_committed": blocks_committed,
            "committed_token_ids": committed_tokens,
            "committed_text": self.tokenizer.decode(committed_tokens, skip_special_tokens=True),
            "bonus_token": bonus_token,
            "stop_reason": stop_reason,
            "decisions": decisions,
        }
        if self._trace_enabled():
            print(
                f"[PREBUILT TREE DONE] visited={visited_node_ids} blocks={blocks_committed} "
                f"stop_reason={stop_reason} text={summary['committed_text']!r}",
                flush=True,
            )
        return summary

    def generate(self, prompt: str, generation_config: GenerationConfig | None = None) -> GenerationResult:
        gc = generation_config or GenerationConfig(
            max_new_tokens=self.config["generation"].get("max_new_tokens", 128), seed=self.config.get("seed", 42)
        )
        encoded = self.tokenizer(prompt, return_tensors="pt")["input_ids"].reshape(-1).long()
        prefix = encoded.clone()
        generated: list[int] = []
        rounds = []
        started = time.perf_counter()
        round_idx = 0
        eos_ids = self.tokenizer.eos_token_id
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        while len(generated) < gc.max_new_tokens and not (generated and generated[-1] in eos_ids):
            remaining = gc.max_new_tokens - len(generated)
            if self._trace_enabled():
                print(f"\n[ROUND {round_idx}] prefix_tokens={prefix.numel()} remaining_tokens={remaining}")
            if remaining < self.drafter.block_size:
                token = self._target_token(prefix.reshape(1, -1))
                generated.append(token)
                prefix = torch.cat((prefix, torch.tensor([token])))
                continue
            draft_start = time.perf_counter()
            tree = self.tree_builder.build(prefix)
            draft_ms = (time.perf_counter() - draft_start) * 1000
            if self._trace_enabled():
                print(f"[DRAFT TREE] nodes={len(tree.nodes)} widths={tree.widths()} build_ms={draft_ms:.2f}")
            if not tree.root.candidate_set:
                if self.config["generation"].get("fallback_mode") != "target_one_token":
                    raise RuntimeError("Drafter returned an empty tree")
                token = self._target_token(prefix.reshape(1, -1))
                generated.append(token)
                prefix = torch.cat((prefix, torch.tensor([token])))
                continue
            node = tree.root
            round_log = {
                "round": round_idx, "prefix_length": int(prefix.numel()), "tree_nodes": len(tree.nodes),
                "tree_widths": tree.widths(), "draft_time_ms": draft_ms, "verify_time_ms": 0.0,
                "node_results": [], "bonus_token_used": False, "committed_tokens": 0,
            }
            reached_expanded_leaf = False
            while node.candidate_set and remaining >= self.drafter.block_size:
                verify_start = time.perf_counter()
                result, diagnostics, child = self._verify_node(tree, node, prefix.reshape(1, -1))
                round_log["verify_time_ms"] += (time.perf_counter() - verify_start) * 1000
                block = node.candidate_set[result.committed_candidate_index].token_ids
                generated.extend(block)
                prefix = torch.cat((prefix, torch.tensor(block)))
                remaining -= len(block)
                round_log["committed_tokens"] += len(block)
                round_log["node_results"].append({**result.__dict__, **diagnostics})
                if any(t in eos_ids for t in block):
                    break
                if child is None:
                    break
                node = child
                if not node.candidate_set:
                    reached_expanded_leaf = True
                    break
            if reached_expanded_leaf and remaining > 0 and self.config["sampling"].get("enable_bonus_token", True):
                bonus = self._target_token(prefix.reshape(1, -1))
                generated.append(bonus)
                prefix = torch.cat((prefix, torch.tensor([bonus])))
                round_log["bonus_token_used"] = True
                round_log["bonus_additional_verifier_call"] = True
                round_log["committed_tokens"] += 1
                if self._trace_enabled():
                    print(f"[BONUS] token={bonus} text={self._block_text((bonus,))!r} additional_verifier_call=true")
            if round_log["node_results"]:
                round_log.update({k: v for k, v in round_log["node_results"][0].items() if k in {
                    "candidate_mass_q", "candidate_mass_p", "proposed_block", "accepted", "residual_used",
                    "committed_block", "continued_in_tree"
                }})
            rounds.append(round_log)
            if self._trace_enabled():
                print(
                    f"[ROUND {round_idx} DONE] committed_tokens={round_log['committed_tokens']} "
                    f"bonus={str(round_log['bonus_token_used']).lower()} total_generated={len(generated)}"
                )
            round_idx += 1
        # Trim only after EOS; max length is already enforced at block boundaries.
        if any(t in eos_ids for t in generated):
            generated = generated[: next(i for i, t in enumerate(generated) if t in eos_ids) + 1]
        elapsed = time.perf_counter() - started
        return GenerationResult(
            self.tokenizer.decode(generated, skip_special_tokens=True), generated,
            int(encoded.numel()), len(generated), elapsed, rounds,
        )

    @staticmethod
    def save_jsonl(result: GenerationResult, path: str | Path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for row in result.rounds:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
