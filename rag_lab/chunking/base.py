from abc import ABC, abstractmethod

from rag_lab.documents import Document, Chunk
from rag_lab.registry import Registry

CHUNKERS = Registry("chunker")


class BaseChunker(ABC):
    NEEDS: tuple[str, ...] = ()   # 此 chunker 需要 pipeline 注入的依賴名稱（如 "embedder"/"llm"）

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]: ...

    def chunk_all(self, docs: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(self.chunk(doc))
        for i, c in enumerate(chunks):   # 給全域 id，hybrid/BM25 會用到
            c.id = i
        return chunks
