import uuid

import chromadb

from .base import BaseVectorStore, VECTOR_STORES


@VECTOR_STORES.register("chroma")
class ChromaStore(BaseVectorStore):
    """嵌入式向量資料庫，支援 metadata 過濾與持久化（這裡用 in-memory）。
    要持久化可改 chromadb.PersistentClient(path="./chroma_db")"""

    def __init__(self, **_):
        self.client = chromadb.EphemeralClient()
        self.col = self.client.create_collection(
            f"rag_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},   # 預設是 L2，記得改 cosine
        )
        self.chunks = []

    def add(self, chunks, vectors):
        start = len(self.chunks)
        self.col.add(
            ids=[str(start + i) for i in range(len(chunks))],
            embeddings=vectors,
            documents=[c.text for c in chunks],
        )
        self.chunks.extend(chunks)

    def search(self, vector, k):
        res = self.col.query(query_embeddings=[vector],
                             n_results=min(k, len(self.chunks)))
        # Chroma 回傳的是「距離」，cosine distance = 1 - similarity
        return [(self.chunks[int(id_)], 1 - dist)
                for id_, dist in zip(res["ids"][0], res["distances"][0])]
