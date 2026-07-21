from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


WIDTHS = [[5, 10, 6, 3, 1], [5, 15, 8, 4, 2, 1], [5, 20, 10, 5, 2, 1], [5, 25, 15, 8, 4, 2, 1]]


def main():
    p = argparse.ArgumentParser(description="Emit and optionally execute the required ablation matrix")
    p.add_argument("--prompt", action="append", default=["Explain why the sky appears blue."])
    p.add_argument("--output-dir", default="outputs/benchmark")
    p.add_argument("--run", action="store_true", help="Run jobs; without this flag only write the manifest")
    p.add_argument("--config", default="configs/colab_balanced.yaml")
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for prompt, width, block, count, bonus, proposal in itertools.product(
        args.prompt, WIDTHS, [2, 4, 5], [3, 5, 8], [False, True], ["top1_q", "sample_q"]
    ):
        jobs.append({"prompt": prompt, "width_schedule": width, "block_size": block,
                     "candidate_count": count, "bonus_token": bonus, "proposal_mode": proposal})
    (out / "manifest.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    rows = []
    if args.run:
        baselines = {}
        for prompt_index, prompt in enumerate(args.prompt):
            metrics_path = out / f"target-metrics-{prompt_index}.json"
            command = [sys.executable, "scripts/target_only.py", "--prompt", prompt,
                       "--metrics-output", str(metrics_path)]
            proc = subprocess.run(command, text=True, capture_output=True)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                raise SystemExit(f"Target-only baseline failed for prompt {prompt_index}")
            baselines[prompt] = json.loads(metrics_path.read_text(encoding="utf-8"))
        # Each job is deliberately isolated so peak GPU memory and failures are attributable.
        for i, job in enumerate(jobs):
            metrics_path = out / f"metrics-{i}.json"
            command = [sys.executable, "scripts/run_generation.py", "--config", args.config,
                       "--prompt", job["prompt"], "--output", str(out / f"rounds-{i}.jsonl"),
                       "--metrics-output", str(metrics_path),
                       "--block-size", str(job["block_size"]),
                       "--candidate-count", str(job["candidate_count"]),
                       "--width-schedule", ",".join(map(str, job["width_schedule"])),
                       "--proposal-mode", job["proposal_mode"]]
            if not job["bonus_token"]: command.append("--disable-bonus-token")
            started = time.perf_counter()
            proc = subprocess.run(command, text=True, capture_output=True)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            baseline_tps = baselines[job["prompt"]]["tokens_per_second"]
            row = {**job, **metrics, "wall_seconds": time.perf_counter() - started,
                   "return_code": proc.returncode,
                   "target_only_tokens_per_second": baseline_tps,
                   "speedup_vs_target_only": metrics.get("tokens_per_second", 0) / baseline_tps}
            rows.append(row)
            with (out / "results.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
    fields = list(rows[0]) if rows else list(jobs[0])
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows or jobs)
    print(f"Wrote {len(jobs)} benchmark jobs to {out}")


if __name__ == "__main__":
    main()
