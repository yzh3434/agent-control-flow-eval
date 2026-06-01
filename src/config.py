import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    model: str
    base_url: str
    sample_size: int
    max_rounds: int
    reflexion_trials: int
    concurrency: int
    easy_max_steps: int
    hard_min_steps: int
    temperature: float
    request_timeout: int
    api_key: str


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines into os.environ (no overwrite)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_config(path: str = "config.yaml") -> Config:
    _load_dotenv()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        model=data["model"],
        base_url=data["base_url"],
        sample_size=int(data["sample_size"]),
        max_rounds=int(data["max_rounds"]),
        reflexion_trials=int(data["reflexion_trials"]),
        concurrency=int(data["concurrency"]),
        easy_max_steps=int(data["easy_max_steps"]),
        hard_min_steps=int(data["hard_min_steps"]),
        temperature=float(data["temperature"]),
        request_timeout=int(data["request_timeout"]),
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    )
