import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class Step:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class Trajectory:
    task_id: str
    controller: str
    difficulty: str
    steps: List[Step]
    final_answer: Optional[str]
    success: bool
    num_rounds: int
    prompt_tokens: int
    completion_tokens: int
    api_calls: int
    trials: int = 1
    reflections: List[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def save_trajectory(traj: Trajectory, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(traj), f, ensure_ascii=False, indent=2)


def load_trajectory(path: str) -> Trajectory:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["steps"] = [Step(**s) for s in data["steps"]]
    return Trajectory(**data)
