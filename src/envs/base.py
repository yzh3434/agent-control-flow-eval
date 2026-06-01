from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from src.tools.base import Tool


@dataclass
class Task:
    id: str
    question: str
    answer: str
    difficulty: str


class Environment(ABC):
    name: str

    @abstractmethod
    def load_tasks(self, difficulty: str) -> List[Task]:
        ...

    @abstractmethod
    def grade(self, task: Task, predicted: str) -> bool:
        ...

    def score(self, task: Task, predicted: str) -> float:
        """Graded score in [0, 1]. Defaults to exact match (1.0/0.0);
        environments with partial credit (e.g. QA F1) override this."""
        return 1.0 if self.grade(task, predicted) else 0.0

    @abstractmethod
    def tools(self, task: Task) -> List[Tool]:
        ...
