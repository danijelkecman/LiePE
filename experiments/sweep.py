from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import subprocess
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="copy_delay")
    parser.add_argument("--encodings", nargs="+", default=["nope", "rope", "alibi", "decay", "damped_rope", "jordan"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--output", default="outputs/sweep_results.csv")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    rows = []
    for enc in args.encodings:
        checkpoint = f"outputs/{args.task}_{enc}.pt"
        cmd = [
            "python", "experiments/train.py",
            "--task", args.task,
            "--encoding", enc,
            "--steps", str(args.steps),
            "--seq-len", str(args.seq_len),
            "--output", checkpoint,
        ]
        print("RUN", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            rows.append({"encoding": enc, "status": "failed"})
        else:
            rows.append({"encoding": enc, "status": "ok", "checkpoint": checkpoint})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
