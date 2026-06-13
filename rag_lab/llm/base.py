from abc import ABC, abstractmethod

from rag_lab.registry import Registry

LLMS = Registry("llm")


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str: ...

    def stream(self, prompt: str, system: str = ""):
        # 預設退化成一次回傳；子類可覆寫成真正的 streaming
        yield self.generate(prompt, system)

    @staticmethod
    def _messages(prompt: str, system: str) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs
