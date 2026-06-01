import json
from src.llm import MockLLMClient
from src.envs.gsm8k import GSM8KEnv
from src.controllers.direct import DirectController
from src.eval.runner import run_controller


def _env(tmp_path):
    rows = [
        {"question": "2+2?", "answer": "x <<2+2=4>> #### 4"},
        {"question": "3+3?", "answer": "x <<3+3=6>> #### 6"},
    ]
    f = tmp_path / "test.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return GSM8KEnv(str(f), easy_max_steps=2, hard_min_steps=5)


def test_runner_runs_all_tasks_and_writes_files(tmp_path):
    env = _env(tmp_path)
    results_dir = tmp_path / "results"
    client = MockLLMClient(responses=["The answer is 4", "The answer is 6"])
    ctrl = DirectController(client)
    trajs = run_controller(ctrl, env, "easy", sample_size=2,
                           results_dir=str(results_dir), concurrency=1)
    assert len(trajs) == 2
    files = list((results_dir / "gsm8k" / "direct" / "easy").glob("*.json"))
    assert len(files) == 2


def test_runner_resumes_and_skips_completed(tmp_path):
    env = _env(tmp_path)
    results_dir = tmp_path / "results"
    # First run completes task 0 only (one response available).
    client1 = MockLLMClient(responses=["The answer is 4"])
    run_controller(DirectController(client1), env, "easy", sample_size=1,
                   results_dir=str(results_dir), concurrency=1)
    # Second run over both tasks should only call the client for task 1.
    client2 = MockLLMClient(responses=["The answer is 6"])
    trajs = run_controller(DirectController(client2), env, "easy", sample_size=2,
                           results_dir=str(results_dir), concurrency=1)
    assert len(trajs) == 2          # returns all (loaded + new)
    assert client2.call_count == 1  # only the uncompleted task hit the API
