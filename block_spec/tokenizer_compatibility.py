from __future__ import annotations

import json
from pathlib import Path

from .types import TokenizerCompatibilityReport


SPECIAL_NAMES = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")


def validate_tokenizer_compatibility(
    drafter_tokenizer,
    verifier_tokenizer,
    *,
    require_compatibility: bool = True,
    report_path: str | Path | None = None,
    allowed_drafter_only_tokens: set[str] | None = None,
) -> TokenizerCompatibilityReport:
    dv, vv = drafter_tokenizer.get_vocab(), verifier_tokenizer.get_vocab()
    allowed = allowed_drafter_only_tokens or {"<|mask|>", "<|MASK|>", "[MASK]", "|<MASK>|"}
    mismatches = []
    for token in sorted(set(dv) | set(vv)):
        if token in allowed and token not in vv:
            continue
        if dv.get(token) != vv.get(token):
            mismatches.append({"token": token, "drafter_id": dv.get(token), "verifier_id": vv.get(token)})
            if len(mismatches) >= 100:
                break
    ds = {n: getattr(drafter_tokenizer, n, None) for n in SPECIAL_NAMES}
    vs = {n: getattr(verifier_tokenizer, n, None) for n in SPECIAL_NAMES}
    special_ok = all(ds[n] == vs[n] for n in SPECIAL_NAMES if ds[n] is not None and vs[n] is not None)
    compatible = not mismatches and special_ok
    reason = "shared token strings map to identical IDs" if compatible else "vocabulary or special-token ID mismatch"
    report = TokenizerCompatibilityReport(compatible, reason, len(dv), len(vv), mismatches, ds, vs)
    if report_path is not None:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    if require_compatibility and not compatible:
        raise ValueError(f"Tokenizer compatibility check failed: {reason}; see {report_path or 'report'}")
    return report


def experimental_text_bridge(block_ids, drafter_tokenizer, verifier_tokenizer) -> tuple[str, list[int]]:
    """Decode/retokenize helper for diagnostics only.

    The returned verifier IDs are not suitable for direct p/q comparison unless
    an experiment defines a principled common candidate space.
    """
    text = drafter_tokenizer.decode(list(block_ids), skip_special_tokens=False)
    verifier_ids = verifier_tokenizer(text, add_special_tokens=False)["input_ids"]
    if verifier_ids and isinstance(verifier_ids[0], list):
        verifier_ids = verifier_ids[0]
    return text, [int(x) for x in verifier_ids]
