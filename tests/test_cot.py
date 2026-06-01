import json
from src.llm import MockLLMClient
from src.envs.gsm8k import GSM8KEnv
from src.controllers.cot import CoTController


def _env(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text(json.dumps({"question": "2+2?", "answer": "x <<2+2=4>> #### 4"}),
                 encoding="utf-8")
    return GSM8KEnv(str(f), easy_max_steps=2, hard_min_steps=5)


def test_cot_controller_grades_last_number_after_reasoning(tmp_path):
    env = _env(tmp_path)
    task = env.load_tasks("easy")[0]
    client = MockLLMClient(responses=["First 2, then add 2, so step by step the answer is 4"])
    traj = CoTController(client).run(task, env)
    assert traj.controller == "cot"
    assert traj.success is True
    assert traj.api_calls == 1
    # the prompt must actually ask for step-by-step reasoning
    assert "step by step" in client.calls[0][0]["content"].lower()
