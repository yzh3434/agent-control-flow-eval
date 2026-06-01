from src.eval.trajectory import Trajectory
from src.eval.metrics import aggregate, reflexion_curve


def _traj(success, rounds, tokens, calls, trials=1):
    return Trajectory(
        task_id="t", controller="react", difficulty="easy", steps=[],
        final_answer="x", success=success, num_rounds=rounds,
        prompt_tokens=tokens, completion_tokens=0, api_calls=calls, trials=trials,
    )


def test_aggregate_computes_rates_and_averages():
    trajs = [_traj(True, 2, 100, 2), _traj(False, 4, 200, 4)]
    m = aggregate(trajs)
    assert m["n"] == 2
    assert m["success_rate"] == 0.5
    assert m["avg_rounds"] == 3.0
    assert m["avg_tokens"] == 150.0
    assert m["avg_calls"] == 3.0


def test_reflexion_curve_cumulative_success_by_trial():
    # task A succeeds on trial 1, task B on trial 3, task C never
    trajs = [_traj(True, 1, 10, 1, trials=1),
             _traj(True, 1, 10, 1, trials=3),
             _traj(False, 1, 10, 1, trials=3)]
    curve = reflexion_curve(trajs, max_trials=3)
    assert curve == [1/3, 1/3, 2/3]
