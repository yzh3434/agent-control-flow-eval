import json
from src.llm import MockLLMClient
from src.envs.gsm8k import GSM8KEnv
from src.controllers.direct import DirectController


def _env(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text(json.dumps({"question": "2+2?", "answer": "x <<2+2=4>> #### 4"}),
                 encoding="utf-8")
    return GSM8KEnv(str(f), easy_max_steps=2, hard_min_steps=5)


def test_direct_controller_succeeds_and_records_one_call(tmp_path):
    env = _env(tmp_path)
    task = env.load_tasks("easy")[0]
    client = MockLLMClient(responses=["The answer is 4"])
    traj = DirectController(client).run(task, env)
    assert traj.controller == "direct"
    assert traj.success is True
    assert traj.final_answer == "The answer is 4"
    assert traj.api_calls == 1
    assert traj.num_rounds == 1
