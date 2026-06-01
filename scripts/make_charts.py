"""Render comparison charts from saved trajectories into figures/.

Produces, per environment:
  - <env>_success_by_difficulty.png   grouped bars of success rate (EM)
  - <env>_f1_by_difficulty.png        grouped bars of token-F1 (HotpotQA only)
  - <env>_tokens_by_difficulty.png    grouped bars of avg tokens (cost)
  - <env>_radar.png                   multi-metric capability radar (hardest tier)

Charts are written to figures/ (committed, so they can be embedded in the README
and the report). Run after scripts/run_experiment.py has populated results/.
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.trajectory import load_trajectory
from src.eval.metrics import aggregate

CONTROLLERS = ["direct", "cot", "react", "reflexion"]
ENV_DIFFICULTIES = {
    "gsm8k": ["easy", "hard"],
    "hotpotqa": ["easy", "medium", "hard"],
}
RESULTS_DIR = "results"
FIG_DIR = "figures"
COLORS = {"direct": "#9e9e9e", "cot": "#42a5f5",
          "react": "#66bb6a", "reflexion": "#ef5350"}


def load_group(env, controller, difficulty):
    pattern = os.path.join(RESULTS_DIR, env, controller, difficulty, "*.json")
    return [load_trajectory(p) for p in glob.glob(pattern)]


def collect(env, difficulties):
    return {c: {d: aggregate(load_group(env, c, d)) for d in difficulties}
            for c in CONTROLLERS}


def _grouped_bar(data, difficulties, metric, title, ylabel, outpath):
    x = np.arange(len(difficulties))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, c in enumerate(CONTROLLERS):
        vals = [data[c][d][metric] for d in difficulties]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=c, color=COLORS[c])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")


def _radar(data, difficulty, env, outpath):
    """Multi-metric capability radar at one difficulty; one polygon per controller."""
    metrics = [("success_rate", "EM")]
    if env == "hotpotqa":
        metrics.append(("avg_f1", "F1"))
    # efficiency axes (higher = cheaper), normalized across controllers
    max_tok = max(data[c][difficulty]["avg_tokens"] for c in CONTROLLERS) or 1.0
    max_rnd = max(data[c][difficulty]["avg_rounds"] for c in CONTROLLERS) or 1.0

    def values(c):
        m = data[c][difficulty]
        vals = [m[key] for key, _ in metrics]
        vals.append(1 - m["avg_tokens"] / max_tok)   # token efficiency
        vals.append(1 - m["avg_rounds"] / max_rnd)   # round efficiency
        return vals

    labels = [lbl for _, lbl in metrics] + ["Token eff.", "Round eff."]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for c in CONTROLLERS:
        vals = values(c)
        vals += vals[:1]
        ax.plot(angles, vals, label=c, color=COLORS[c], linewidth=2)
        ax.fill(angles, vals, color=COLORS[c], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"{env} capability radar ({difficulty})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="gsm8k", choices=list(ENV_DIFFICULTIES))
    parser.add_argument("--difficulties", nargs="+", default=None)
    args = parser.parse_args()
    difficulties = args.difficulties or ENV_DIFFICULTIES[args.env]

    os.makedirs(FIG_DIR, exist_ok=True)
    data = collect(args.env, difficulties)

    _grouped_bar(data, difficulties, "success_rate",
                 f"{args.env}: success rate (EM) by difficulty", "success rate",
                 os.path.join(FIG_DIR, f"{args.env}_success_by_difficulty.png"))
    if args.env == "hotpotqa":
        _grouped_bar(data, difficulties, "avg_f1",
                     f"{args.env}: token-F1 by difficulty", "F1",
                     os.path.join(FIG_DIR, f"{args.env}_f1_by_difficulty.png"))
    _grouped_bar(data, difficulties, "avg_tokens",
                 f"{args.env}: avg tokens per question by difficulty", "avg tokens",
                 os.path.join(FIG_DIR, f"{args.env}_tokens_by_difficulty.png"))
    _radar(data, difficulties[-1], args.env,
           os.path.join(FIG_DIR, f"{args.env}_radar.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
