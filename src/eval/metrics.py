from typing import List

from src.eval.trajectory import Trajectory


def aggregate(trajectories: List[Trajectory]) -> dict:
    n = len(trajectories)
    if n == 0:
        return {"n": 0, "success_rate": 0.0, "avg_rounds": 0.0,
                "avg_tokens": 0.0, "avg_calls": 0.0}
    successes = sum(1 for t in trajectories if t.success)
    return {
        "n": n,
        "success_rate": successes / n,
        "avg_rounds": sum(t.num_rounds for t in trajectories) / n,
        "avg_tokens": sum(t.total_tokens for t in trajectories) / n,
        "avg_calls": sum(t.api_calls for t in trajectories) / n,
    }


def reflexion_curve(trajectories: List[Trajectory], max_trials: int) -> List[float]:
    """Cumulative success rate after k trials, for k = 1..max_trials.

    A task that succeeded is counted from its `trials` value onward.
    """
    n = len(trajectories)
    if n == 0:
        return [0.0] * max_trials
    curve = []
    for k in range(1, max_trials + 1):
        solved = sum(1 for t in trajectories if t.success and t.trials <= k)
        curve.append(solved / n)
    return curve
