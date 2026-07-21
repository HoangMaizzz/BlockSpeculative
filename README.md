# Block-level speculative decoding: Fast-dLLM v2 + Qwen2.5-7B

This project implements fixed-length block speculative decoding with the real `Efficient-Large-Model/Fast_dLLM_v2_1.5B` remote-code model as drafter and a locally stored Qwen causal LM as verifier. It includes heap Top-K joint construction, an asymmetric draft tree, batched AR scoring, normalized truncated acceptance/residual sampling, an optional AR bonus token, tests, diagnostics, and a Colab runner.

The original `load_model.ipynb` downloaded both checkpoints by model ID. Production scripts default to local-only loading and will not download models unless `--allow-model-download` is explicitly supplied.

## Local setup and tests

```bash
python -m pip install -r requirements-colab.txt
python -m pytest
python scripts/benchmark.py
```

The last command writes the full 144-job ablation manifest and CSV without loading weights. Add `--run` on a suitable GPU to execute it.

## Colab run with existing models

```bash
export DRAFTER_MODEL_PATH=/content/models/fast-dllm-1.5b
export VERIFIER_MODEL_PATH=/content/models/qwen-7b

python scripts/inspect_models.py \
  --drafter-model-path "$DRAFTER_MODEL_PATH" \
  --verifier-model-path "$VERIFIER_MODEL_PATH"

python scripts/run_generation.py \
  --config configs/colab_balanced.yaml \
  --drafter-model-path "$DRAFTER_MODEL_PATH" \
  --verifier-model-path "$VERIFIER_MODEL_PATH" \
  --prompt "Solve: Natalia sold 48 clips in April and half as many in May." \
  --max-new-tokens 128
```

For the target-only comparison:

```bash
python scripts/target_only.py \
  --verifier-model-path "$VERIFIER_MODEL_PATH" \
  --prompt "Solve: Natalia sold 48 clips in April and half as many in May." \
  --max-new-tokens 128
```

Start with `configs/colab_light.yaml` on constrained GPUs. `device_map: auto` permits Transformers offload; verifier candidates are automatically retried in smaller chunks after CUDA OOM. Quantization is deliberately not enabled.

## Configuration and outputs

CLI model paths override YAML paths; `DRAFTER_MODEL_PATH` and `VERIFIER_MODEL_PATH` are used when CLI paths are absent. Standard Hugging Face `HF_HOME` and `TRANSFORMERS_CACHE` are honored by Transformers. Round JSONL and the tokenizer report are written beneath `outputs/`, which is ignored by Git.

Verification tracing is enabled by default. For every node the terminal prints the shared
candidate table (`q_S`, `p_S`, residual probability and block text), proposed index,
acceptance probability `alpha`, uniform sample `u`, ACCEPT/REJECT verdict, residual sample,
committed block, and whether verification continues to a child. Disable it with:

```bash
python scripts/run_generation.py ... --disable-verification-trace
```

The default schedule is `[5, 20, 10, 5, 2, 1]`. The second depth first grants two strong children per depth-1 parent, then fills remaining positions globally. A candidate remains in its parent's residual support even when it has no child node.

See [repository inspection](docs/repository_inspection.md), [algorithm](docs/algorithm.md), [Colab guide](docs/colab_usage.md), and [limitations](docs/limitations.md).

## Git/GitHub workflow

This supplied directory did not contain `.git`, so no branch or commits were created automatically. From the repository root:

```bash
git init                         # only if this is still not a Git repository
git checkout -b feature/block-spec-fastdllm-qwen
git status
git add .gitignore README.md pyproject.toml requirements-colab.txt block_spec configs scripts tests docs notebooks load_model.ipynb
git commit -m "Implement Fast-dLLM Qwen block speculative decoding"
git remote add origin https://github.com/<USER>/<REPOSITORY>.git  # if needed
git push -u origin feature/block-spec-fastdllm-qwen
```

No model weights, caches, output logs, or secrets should be committed.
