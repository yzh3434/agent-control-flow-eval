import json
from src.llm import MockLLMClient
from src.envs.gsm8k import GSM8KEnv
from src.controllers.reflexion import ReflexionController


def _env(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text(json.dumps({"question": "What is 21*2?", "answer": "x <<21*2=42>> #### 42"}),
                 encoding="utf-8")
    return GSM8KEnv(str(f), easy_max_steps=2, hard_min_steps=5)


def test_reflexion_retries_after_failure_then_succeeds(tmp_path):
    env = _env(tmp_path)
    task = env.load_tasks("easy")[0]
    # Trial 1: wrong Finish -> fail. Then a reflection. Trial 2: correct Finish.
    client = MockLLMClient(responses=[
        "Thought: guess.\nAction: Finish[99]",     # trial 1 (fails)
        "I miscalculated; I should multiply 21 by 2.",  # reflection
        "Thought: 21*2=42.\nAction: Finish[42]",   # trial 2 (succeeds)
    ])
    traj = ReflexionController(client, max_rounds=7, max_trials=3).run(task, env)
    assert traj.controller == "reflexion"
    assert traj.success is True
    assert traj.trials == 2
    assert len(traj.reflections) == 1
    assert traj.api_calls == 3  # 2 react calls + 1 reflection call


def test_reflexion_stops_when_first_trial_succeeds(tmp_path):
    env = _env(tmp_path)
    task = env.load_tasks("easy")[0]
    client = MockLLMClient(responses=["Thought: 21*2=42.\nAction: Finish[42]"])
    traj = ReflexionController(client, max_rounds=7, max_trials=3).run(task, env)
    assert traj.success is True
    assert traj.trials == 1
    assert traj.reflections == []
