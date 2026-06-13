from abc import ABC, abstractmethod

from rag_lab.documents import Chunk
from rag_lab.registry import Registry

RETRIEVERS = Registry("retriever")


class BaseRetriever(ABC):
    """所有 retriever 共用的依賴（用不到的留 None 即可）"""

    def __init__(self, embedder=None, store=None, chunks=None,
                 llm=None, top_k: int = 5, **_):
        self.embedder = embedder
        self.store = store
        self.chunks = chunks or []
        self.llm = llm
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, query: str) -> list[tuple[Chunk, float]]: ...
