from src.eval.trajectory import Step, Trajectory, save_trajectory, load_trajectory


def make_traj():
    return Trajectory(
        task_id="gsm8k-easy-0", controller="react", difficulty="easy",
        steps=[Step(thought="t", action="calculator", action_input="1+1", observation="2")],
        final_answer="2", success=True, num_rounds=1,
        prompt_tokens=10, completion_tokens=5, api_calls=1,
        trials=1, reflections=[],
    )


def test_total_tokens():
    assert make_traj().total_tokens == 15


def test_round_trip_json(tmp_path):
    traj = make_traj()
    path = tmp_path / "t.json"
    save_trajectory(traj, str(path))
    loaded = load_trajectory(str(path))
    assert loaded == traj
    assert loaded.steps[0].action == "calculator"
