import json
import re
from typing import List, Optional

from src.envs.base import Environment, Task
from src.tools.base import Tool
from src.tools.calculator import make_calculator

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_gold_answer(answer_field: str) -> str:
    """Return the text after GSM8K's '####' marker, stripped of commas."""
    after = answer_field.split("####")[-1].strip()
    return after.replace(",", "")


def count_steps(answer_field: str) -> int:
    """Count GSM8K calculator annotations <<...>> as a proxy for reasoning steps."""
    return answer_field.count("<<")


def classify_difficulty(steps: int, easy_max: int, hard_min: int) -> Optional[str]:
    if steps <= easy_max:
        return "easy"
    if steps >= hard_min:
        return "hard"
    return None  # medium band dropped to keep tiers well separated


def extract_pred_number(text: str) -> Optional[float]:
    """Extract the last number from free-form model output."""
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    matches = [m for m in matches if re.search(r"\d", m)]
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def numbers_match(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(float(a) - float(b)) <= tol


class GSM8KEnv(Environment):
    name = "gsm8k"

    def __init__(self, data_path: str, easy_max_steps: int, hard_min_steps: int):
        self.data_path = data_path
        self.easy_max_steps = easy_max_steps
        self.hard_min_steps = hard_min_steps

    def _iter_rows(self):
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def load_tasks(self, difficulty: str) -> List[Task]:
        tasks = []
        for i, row in enumerate(self._iter_rows()):
            steps = count_steps(row["answer"])
            tier = classify_difficulty(steps, self.easy_max_steps, self.hard_min_steps)
            if tier == difficulty:
                tasks.append(Task(
                    id=f"gsm8k-{difficulty}-{i}",
                    question=row["question"],
                    answer=extract_gold_answer(row["answer"]),
                    difficulty=difficulty,
                ))
        return tasks

    def grade(self, task: Task, predicted: str) -> bool:
        pred = extract_pred_number(predicted)
        if pred is None:
            return False
        try:
            gold = float(task.answer)
        except ValueError:
            return False
        return numbers_match(pred, gold)

    def tools(self, task: Task) -> List[Tool]:
        return [make_calculator()]
