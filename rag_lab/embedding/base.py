from abc import ABC, abstractmethod

from rag_lab.registry import Registry

EMBEDDERS = Registry("embedder")


class BaseEmbedder(ABC):
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        # 預設查詢和文件用同一種方式編碼；BGE 系列會覆寫（查詢要加 prefix）
        return self.embed_documents([text])[0]
