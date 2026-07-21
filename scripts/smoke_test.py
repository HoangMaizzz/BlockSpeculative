from __future__ import annotations

import os
import subprocess
import sys


cmd = [sys.executable, "scripts/run_generation.py", "--config", "configs/colab_light.yaml",
       "--drafter-model-path", os.environ["DRAFTER_MODEL_PATH"],
       "--verifier-model-path", os.environ["VERIFIER_MODEL_PATH"],
       "--prompt", "Write one short sentence about the Moon.", "--max-new-tokens", "16"]
raise SystemExit(subprocess.call(cmd))
