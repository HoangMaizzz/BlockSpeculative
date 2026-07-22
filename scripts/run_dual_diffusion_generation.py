"""Convenience entry point for Fast-dLLM drafter + Fast-dLLM verifier."""

from __future__ import annotations

import sys

from run_union_generation import main


if __name__ == "__main__":
    if "--verifier-backend" not in sys.argv:
        sys.argv.extend(("--verifier-backend", "fast_dllm"))
    main()
