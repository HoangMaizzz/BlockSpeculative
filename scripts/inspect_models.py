from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from block_spec.tokenizer_compatibility import validate_tokenizer_compatibility
from block_spec.verifier_loader import load_local_causal_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter-model-path", required=True)
    p.add_argument("--verifier-model-path", required=True)
    p.add_argument("--allow-model-download", action="store_true")
    args = p.parse_args()
    dm, dt = load_local_causal_model(args.drafter_model_path, allow_model_download=args.allow_model_download)
    vm, vt = load_local_causal_model(args.verifier_model_path, allow_model_download=args.allow_model_download)
    report = validate_tokenizer_compatibility(dt, vt, require_compatibility=False,
                                              report_path="outputs/tokenizer_compatibility.json")
    print(report.to_dict())
    print("Drafter forward:", dm.forward.__doc__ or "(no docstring)")
    print("Verifier forward:", vm.forward.__doc__ or "(no docstring)")


if __name__ == "__main__":
    main()

