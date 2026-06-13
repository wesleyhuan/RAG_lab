from abc import ABC, abstractmethod

from rag_lab.documents import Chunk
from rag_lab.registry import Registry

VECTOR_STORES = Registry("vector_store")


class BaseVectorStore(ABC):
    """統一介面：add 存入向量，search 回傳 (Chunk, 相似度) 由高到低"""

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: list[list[float]]): ...

    @abstractmethod
    def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]: ...
