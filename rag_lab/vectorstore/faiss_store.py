import faiss
import numpy as np

from .base import BaseVectorStore, VECTOR_STORES


@VECTOR_STORES.register("faiss")
class FaissStore(BaseVectorStore):
    """IndexFlatIP（內積）+ L2 正規化向量 = cosine 相似度。
    純記憶體、暴力搜尋，幾萬筆以內最快最簡單。"""

    def __init__(self, **_):
        self.index = None
        self.chunks = []

    def add(self, chunks, vectors):
        vecs = np.array(vectors, dtype="float32")
        faiss.normalize_L2(vecs)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, vector, k):
        q = np.array([vector], dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s))
                for s, i in zip(scores[0], ids[0]) if i != -1]
