# Colab usage

Use a GPU runtime. Clone the branch, install `requirements-colab.txt`, and point both environment variables at already downloaded model directories. `local_files_only=True` is the default; missing directories fail clearly rather than downloading weights.

Run `scripts/inspect_models.py` before generation. The 1.5B drafter plus 7B verifier can exceed smaller Colab GPUs in FP16/BF16. Device-map CPU offload and verifier candidate chunking are supported; automatic quantization is intentionally disabled.

The bundled notebook contains diagnostics, clone/install placeholders, path assertions, inspection, smoke test, balanced run, and benchmark cells.
