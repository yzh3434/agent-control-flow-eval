"""Aggregate saved trajectories into the main comparison table (markdown + CSV)."""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.trajectory import load_trajectory
from src.eval.metrics import aggregate, reflexion_curve

RESULTS_DIR = "results"
ENV = "gsm8k"
CONTROLLERS = ["direct", "cot", "react", "reflexion"]
DIFFICULTIES = ["easy", "hard"]


def load_group(controller, difficulty):
    pattern = os.path.join(RESULTS_DIR, ENV, controller, difficulty, "*.json")
    return [load_trajectory(p) for p in glob.glob(pattern)]


def main():
    rows = []
    for diff in DIFFICULTIES:
        for metric in ["success_rate", "avg_rounds", "avg_tokens"]:
            row = {"difficulty": diff, "metric": metric}
            for ctrl in CONTROLLERS:
                trajs = load_group(ctrl, diff)
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
    for diff in DIFFICULTIES:
        trajs = load_group("reflexion", diff)
        if trajs:
            curve = reflexion_curve(trajs, max_trials=max(t.trials for t in trajs))
            print(f"\nReflexion success vs trial ({diff}): "
                  + ", ".join(f"trial{i+1}={v:.2%}" for i, v in enumerate(curve)))

    # CSV
    out_csv = os.path.join(RESULTS_DIR, "report_table.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["difficulty", "metric"] + CONTROLLERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
