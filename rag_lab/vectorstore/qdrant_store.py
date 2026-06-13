import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .base import BaseVectorStore, VECTOR_STORES


@VECTOR_STORES.register("qdrant")
class QdrantStore(BaseVectorStore):
    """這裡用 :memory: 模式，不需要 Docker。
    正式環境可改 QdrantClient(url="http://localhost:6333")（docker run qdrant/qdrant）"""

    def __init__(self, **_):
        self.client = QdrantClient(":memory:")
        self.name = f"rag_{uuid.uuid4().hex[:8]}"
        self.chunks = []
        self._created = False

    def add(self, chunks, vectors):
        if not self._created:
            self.client.create_collection(
                self.name,
                vectors_config=VectorParams(size=len(vectors[0]),
                                            distance=Distance.COSINE),
            )
            self._created = True
        start = len(self.chunks)
        points = [PointStruct(id=start + i, vector=v, payload={})
                  for i, v in enumerate(vectors)]
        self.client.upsert(self.name, points)
        self.chunks.extend(chunks)

    def search(self, vector, k):
        hits = self.client.query_points(
            self.name, query=vector, limit=min(k, len(self.chunks))
        ).points
        return [(self.chunks[h.id], h.score) for h in hits]
