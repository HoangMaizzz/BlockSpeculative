# Kaggle: inspect a five-block drafter tree

This mode loads only Fast-dLLM. It does not load Qwen and does not run acceptance,
residual sampling, bonus generation, or subsequent decoding rounds. The output is a
five-level asymmetric tree whose blocks contain three tokens each.

Use a Kaggle GPU session and enable Internet while cloning/installing. Attach the
Fast-dLLM checkpoint as a Kaggle Model/Dataset when offline execution is required.

Locate an attached checkpoint with:

```bash
find /kaggle/input -maxdepth 6 -name config.json
```

Set `DRAFTER_MODEL_PATH` to the directory containing `config.json`, `modeling.py`,
the tokenizer files, and the model weights. Then run:

```bash
python -u scripts/inspect_drafter_tree.py \
  --drafter-model-path "$DRAFTER_MODEL_PATH" \
  --prompt "Natalia sold 48 clips in April and half as many in May. How many did she sell altogether?" \
  --block-size 3 \
  --num-block-candidates 5 \
  --per-position-topk 8 \
  --depth 5 \
  --width-schedule 5,20,10,5,1 \
  --output outputs/drafter_tree_step1.json
```

For an Internet-enabled run without an attached checkpoint, use the Hugging Face model
ID and explicitly allow the download:

```bash
python -u scripts/inspect_drafter_tree.py \
  --drafter-model-path Efficient-Large-Model/Fast_dLLM_v2_1.5B \
  --allow-model-download \
  --prompt "Natalia sold 48 clips in April and half as many in May. How many did she sell altogether?" \
  --block-size 3 --num-block-candidates 5 --per-position-topk 8 \
  --depth 5 --width-schedule 5,20,10,5,1
```

`joint_probability` is the product of the three one-pass masked-position marginals.
It is not the probability of Fast-dLLM's complete iterative unmasking trajectory.
Each `normalized_probability_at_parent` value is normalized only among the five
local candidates at that parent. The JSON report retains all local candidates,
including branches pruned by the global width schedule.
