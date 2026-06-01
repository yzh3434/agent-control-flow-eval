"""Aggregate saved trajectories into the main comparison table (markdown + CSV)."""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.trajectory import load_trajectory
from src.eval.metrics import aggregate, reflexion_curve

RESULTS_DIR = "results"
CONTROLLERS = ["direct", "cot", "react", "reflexion"]
ENV_DIFFICULTIES = {
    "gsm8k": ["easy", "hard"],
    "hotpotqa": ["easy", "medium", "hard"],
}


def load_group(env, controller, difficulty):
    pattern = os.path.join(RESULTS_DIR, env, controller, difficulty, "*.json")
    return [load_trajectory(p) for p in glob.glob(pattern)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="gsm8k", choices=list(ENV_DIFFICULTIES))
    parser.add_argument("--difficulties", nargs="+", default=None)
    args = parser.parse_args()
    difficulties = args.difficulties or ENV_DIFFICULTIES[args.env]

    metrics = ["success_rate", "avg_rounds", "avg_tokens"]
    if args.env == "hotpotqa":
        # QA answers get partial credit, so report token-F1 alongside exact match
        metrics = ["success_rate", "avg_f1", "avg_rounds", "avg_tokens"]

    rows = []
    for diff in difficulties:
        for metric in metrics:
            row = {"difficulty": diff, "metric": metric}
            for ctrl in CONTROLLERS:
                trajs = load_group(args.env, ctrl, diff)
                m = aggregate(trajs)
                row[ctrl] = round(m[metric], 4)
            rows.append(row)

    # Markdown
    header = "| difficulty | metric | " + " | ".join(CONTROLLERS) + " |"
    sep = "|" + "---|" * (2 + len(CONTROLLERS))
    print(header)
    print(sep)
    for r in rows:
        cells = [r["difficulty"], r["metric"]] + [str(r[c]) for c in CONTROLLERS]
        print("| " + " | ".join(cells) + " |")

    # Reflexion curve
    for diff in difficulties:
        trajs = load_group(args.env, "reflexion", diff)
        if trajs:
            curve = reflexion_curve(trajs, max_trials=max(t.trials for t in trajs))
            print(f"\nReflexion success vs trial ({diff}): "
                  + ", ".join(f"trial{i+1}={v:.2%}" for i, v in enumerate(curve)))

    # CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f"report_table_{args.env}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["difficulty", "metric"] + CONTROLLERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
