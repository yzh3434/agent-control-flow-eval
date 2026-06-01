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

    @abstractmethod
    def tools(self) -> List[Tool]:
        ...
