"""HotpotQA environment (distractor setting).

Each question ships with context paragraphs and a native difficulty `level`
(easy/medium/hard). Direct/CoT answer closed-book (no context); ReAct/Reflexion
get search/lookup tools over the question's context. Grading is normalized
Exact Match, the standard HotpotQA answer metric.
"""
import json
import re
import string
from collections import Counter
from typing import List

from src.envs.base import Environment, Task
from src.tools.base import Tool
from src.tools.wiki import build_wiki_tools


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA answer normalization: lowercase, strip punctuation,
    articles, and redundant whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_answer(text: str) -> str:
    """Pull a candidate answer from free-form output (handles 'The answer is X')."""
    matches = list(re.finditer(r"answer\s*(?:is|:)\s*(.+)", text, re.IGNORECASE))
    candidate = matches[-1].group(1) if matches else text
    return candidate.strip().rstrip(".").strip()


def exact_match(predicted: str, gold: str) -> bool:
    return normalize_answer(extract_answer(predicted)) == normalize_answer(gold)


def f1_score(predicted: str, gold: str) -> float:
    """Token-level F1 between the extracted answer and gold (HotpotQA style)."""
    pred_tokens = normalize_answer(extract_answer(predicted)).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class HotpotQAEnv(Environment):
    name = "hotpotqa"

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._context_by_id = {}

    def _load_raw(self):
        with open(self.data_path, encoding="utf-8") as f:
            return json.load(f)

    def load_tasks(self, difficulty: str) -> List[Task]:
        tasks = []
        for row in self._load_raw():
            if row.get("level") != difficulty:
                continue
            task_id = f"hotpotqa-{difficulty}-{row['_id']}"
            self._context_by_id[task_id] = row.get("context", [])
            tasks.append(Task(
                id=task_id,
                question=row["question"],
                answer=row["answer"],
                difficulty=difficulty,
            ))
        return tasks

    def grade(self, task: Task, predicted: str) -> bool:
        if predicted is None:
            return False
        return exact_match(predicted, task.answer)

    def score(self, task: Task, predicted: str) -> float:
        if predicted is None:
            return 0.0
        return f1_score(predicted, task.answer)

    def tools(self, task: Task) -> List[Tool]:
        context = self._context_by_id.get(task.id, [])
        return build_wiki_tools(context)
