from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]

    def run(self, arg: str) -> str:
        return self.func(arg)
