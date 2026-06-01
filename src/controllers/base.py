from abc import ABC, abstractmethod

from src.envs.base import Environment, Task
from src.eval.trajectory import Trajectory


class Controller(ABC):
    name: str

    def __init__(self, client):
        self.client = client

    @abstractmethod
    def run(self, task: Task, env: Environment) -> Trajectory:
        ...
