import json
from src.envs.gsm8k import (
    extract_gold_answer, count_steps, extract_pred_number,
    numbers_match, GSM8KEnv,
)

GOLD = "Janet has 3 apples.\nShe buys 2 <<3+2=5>> more.\n#### 5"


def test_extract_gold_answer():
    assert extract_gold_answer(GOLD) == "5"


def test_count_steps_counts_calc_annotations():
    assert count_steps(GOLD) == 1
    assert count_steps("a <<1+1=2>> b <<2+2=4>> c #### 4") == 2


def test_extract_pred_number_takes_last_number():
    assert extract_pred_number("The answer is 42.") == 42.0
    assert extract_pred_number("Finish[5]") == 5.0
    assert extract_pred_number("no number here") is None
    assert extract_pred_number("result: 1,234 dollars") == 1234.0


def test_numbers_match_with_tolerance():
    assert numbers_match(5.0, 5) is True
    assert numbers_match(5.0001, 5) is True
    assert numbers_match(6.0, 5) is False


def test_load_tasks_stratifies_by_difficulty(tmp_path):
    lines = [
        {"question": "easy q", "answer": "x <<1+1=2>> #### 2"},        # 1 step -> easy
        {"question": "hard q",
         "answer": "a <<1+1=2>> b <<2+2=4>> c <<4+4=8>> d <<8+1=9>> e <<9+1=10>> #### 10"},  # 5 steps -> hard
    ]
    data_file = tmp_path / "test.jsonl"
    data_file.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    env = GSM8KEnv(str(data_file), easy_max_steps=2, hard_min_steps=5)
    easy = env.load_tasks("easy")
    hard = env.load_tasks("hard")
    assert len(easy) == 1 and easy[0].difficulty == "easy"
    assert len(hard) == 1 and hard[0].difficulty == "hard"


def test_grade_uses_numeric_match(tmp_path):
    data_file = tmp_path / "test.jsonl"
    data_file.write_text(json.dumps({"question": "q", "answer": "x <<1+1=2>> #### 2"}),
                         encoding="utf-8")
    env = GSM8KEnv(str(data_file), easy_max_steps=2, hard_min_steps=5)
    task = env.load_tasks("easy")[0]
    assert env.grade(task, "The answer is 2") is True
    assert env.grade(task, "The answer is 3") is False
