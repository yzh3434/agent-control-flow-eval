import json
from src.llm import MockLLMClient
from src.envs.gsm8k import GSM8KEnv
from src.controllers.react import ReActController, parse_action


def _env(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text(json.dumps({"question": "What is 21*2?", "answer": "x <<21*2=42>> #### 42"}),
                 encoding="utf-8")
    return GSM8KEnv(str(f), easy_max_steps=2, hard_min_steps=5)


def test_parse_action_extracts_name_and_input():
    assert parse_action("Thought: I should compute.\nAction: calculator[21*2]") == ("calculator", "21*2")
    assert parse_action("Action: Finish[42]") == ("Finish", "42")
    assert parse_action("no action here") == (None, None)


def test_react_uses_tool_then_finishes(tmp_path):
    env = _env(tmp_path)
    task = env.load_tasks("easy")[0]
    client = MockLLMClient(responses=[
        "Thought: compute it.\nAction: calculator[21*2]",
        "Thought: got 42.\nAction: Finish[42]",
    ])
    traj = ReActController(client, max_rounds=7).run(task, env)
    assert traj.controller == "react"
    assert traj.success is True
    assert traj.final_answer == "42"
    assert traj.num_rounds == 2
    assert traj.api_calls == 2
    assert traj.steps[0].action == "calculator"
    assert traj.steps[0].observation == "42"


def test_react_stops_at_max_rounds_without_finish(tmp_path):
    env = _env(tmp_path)
    task = env.load_tasks("easy")[0]
    client = MockLLMClient(responses=["Thought: hmm.\nAction: calculator[1+1]"] * 3)
    traj = ReActController(client, max_rounds=3).run(task, env)
    assert traj.num_rounds == 3
    assert traj.success is False
