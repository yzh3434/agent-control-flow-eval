import time
from dataclasses import dataclass

import requests


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class MockLLMClient:
    """Deterministic client for tests. Returns scripted responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0
        self.calls = []  # records messages passed, for assertions

    def chat(self, messages):
        self.calls.append(messages)
        text = self._responses[self.call_count]  # IndexError when exhausted
        self.call_count += 1
        # rough token estimate: 1 token per 4 chars
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        return ChatResult(
            text=text,
            prompt_tokens=max(1, prompt_chars // 4),
            completion_tokens=max(1, len(text) // 4),
        )


class DeepSeekClient:
    """DeepSeek chat client (OpenAI-compatible /chat/completions endpoint)."""

    def __init__(self, api_key, model, base_url, temperature=0.0,
                 timeout=60, max_retries=4):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, headers=headers,
                                     timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                return ChatResult(
                    text=data["choices"][0]["message"]["content"],
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
            except (requests.RequestException, KeyError) as err:
                last_err = err
                time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek request failed after retries: {last_err}")
