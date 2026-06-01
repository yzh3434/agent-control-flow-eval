"""Run selected control flows on GSM8K difficulty tiers and save trajectories."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.llm import DeepSeekClient
from src.envs.gsm8k import GSM8KEnv
from src.controllers.direct import DirectController
from src.controllers.cot import CoTController
from src.controllers.react import ReActController
from src.controllers.reflexion import ReflexionController
from src.eval.runner import run_controller
from src.eval.metrics import aggregate

DATA_PATH = os.path.join("data", "gsm8k_test.jsonl")
RESULTS_DIR = "results"


def build_controller(name, client, cfg):
    if name == "direct":
        return DirectController(client)
    if name == "cot":
        return CoTController(client)
    if name == "react":
        return ReActController(client, max_rounds=cfg.max_rounds)
    if name == "reflexion":
        return ReflexionController(client, max_rounds=cfg.max_rounds,
                                   max_trials=cfg.reflexion_trials)
    raise ValueError(f"unknown controller: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", nargs="+",
                        default=["direct", "cot", "react", "reflexion"])
    parser.add_argument("--difficulty", nargs="+", default=["easy", "hard"])
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if not cfg.api_key:
        print("ERROR: DEEPSEEK_API_KEY not set. Copy .env.example to .env and fill it in.")
        return 1
    sample_size = args.sample_size or cfg.sample_size

    client = DeepSeekClient(api_key=cfg.api_key, model=cfg.model,
                            base_url=cfg.base_url, temperature=cfg.temperature,
                            timeout=cfg.request_timeout)
    env = GSM8KEnv(DATA_PATH, easy_max_steps=cfg.easy_max_steps,
                   hard_min_steps=cfg.hard_min_steps)

    for ctrl_name in args.controllers:
        for diff in args.difficulty:
            ctrl = build_controller(ctrl_name, client, cfg)
            print(f"Running {ctrl_name} on {diff} (n<={sample_size}) ...")
            trajs = run_controller(ctrl, env, diff, sample_size=sample_size,
                                   results_dir=RESULTS_DIR, concurrency=cfg.concurrency)
            m = aggregate(trajs)
            print(f"  {ctrl_name}/{diff}: success={m['success_rate']:.2%} "
                  f"avg_rounds={m['avg_rounds']:.1f} avg_tokens={m['avg_tokens']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
