# Colab usage

Use a GPU runtime. Clone the branch, install `requirements-colab.txt`, and point both environment variables at already downloaded model directories. `local_files_only=True` is the default; missing directories fail clearly rather than downloading weights.

Run `scripts/inspect_models.py` before generation. The 1.5B drafter plus 7B verifier can exceed smaller Colab GPUs in FP16/BF16. Device-map CPU offload and verifier candidate chunking are supported; automatic quantization is intentionally disabled.

The bundled notebook contains diagnostics, clone/install placeholders, path assertions, inspection, smoke test, balanced run, and benchmark cells.

`logging.print_verification_trace` and `logging.print_candidate_table` are enabled in the
bundled configurations. The same decisions and random seed are persisted in the round JSONL
file so a run can be audited after Colab finishes.

To inspect one complete asymmetric tree with the shared union support, run
`scripts/verify_drafter_tree.py` with `--candidate-set-mode union_topk`. Each
node keeps its Fast-dLLM marginal table in FP16 CPU memory, while verifier beam
logits are reduced to Top-K immediately. On an L4, the defaults reserve Qwen to
14 GiB of GPU weight placement and start verifier beam batches at 8; lower
`--verifier-beam-batch-size` or `--verifier-max-gpu-memory-gib` if another
notebook allocation is present.

For a complete answer rather than one inspected tree, use
`scripts/run_union_generation.py`. It keeps the 1.5B drafter and a smaller
verifier such as Qwen2.5-3B loaded once, then repeats build, union enrichment,
one-pass tree scoring, and cached residual traversal until EOS, a final-answer
pattern, or the new-token limit.

The default five-level draft width is `[5, 20, 20, 20, 20]`. Selection is
global by cumulative drafter log-probability with no mandatory depth-2 parent
quota, so high-probability parents may contribute more children. The default
budgets are 96 tree nodes and 320 tree tokens, sufficient for all 86 nodes at
block length 3.
