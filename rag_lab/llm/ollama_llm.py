import json

import requests

from .base import BaseLLM, LLMS


@LLMS.register("ollama")
class OllamaLLM(BaseLLM):
    """直接打 Ollama REST API（不裝額外 SDK，看得到底層長什麼樣）。
    注意 num_ctx：Ollama 預設 context 只有 2048/4096，RAG prompt 很容易爆，
    所以這裡明確帶 options 把它撐大。"""

    def __init__(self, model: str = "qwen2.5:7b",
                 base_url: str = "http://localhost:11434",
                 num_ctx: int = 8192, **_):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx

    def _payload(self, prompt, system, stream):
        return {
            "model": self.model,
            "messages": self._messages(prompt, system),
            "stream": stream,
            "options": {"num_ctx": self.num_ctx},
        }

    def generate(self, prompt, system=""):
        r = requests.post(f"{self.base_url}/api/chat",
                          json=self._payload(prompt, system, False), timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def stream(self, prompt, system=""):
        with requests.post(f"{self.base_url}/api/chat",
                           json=self._payload(prompt, system, True),
                           stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content")
                if token:
                    yield token
