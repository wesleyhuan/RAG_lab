import re

import numpy as np

from .base import BaseChunker, CHUNKERS
from rag_lab.documents import Document, Chunk


@CHUNKERS.register("semantic")
class SemanticChunker(BaseChunker):
    """語意切割：把每個句子向量化，計算「相鄰句子」的 cosine 距離，
    距離超過門檻（取所有距離的某個百分位數）就視為主題轉換 -> 斷開。
    優點：chunk 內語意連貫。缺點：建索引時要多跑一次 embedding。"""

    def __init__(self, embedder=None, percentile: int = 80,
                 max_chunk_size: int = 1200, **_):
        if embedder is None:
            raise ValueError("semantic chunker 需要注入 embedder")
        self.embedder = embedder
        self.percentile = percentile
        self.max_chunk_size = max_chunk_size

    @staticmethod
    def _sentences(text: str) -> list[str]:
        sents = re.split(r"(?<=[。！？!?\.])\s*|\n+", text)
        return [s.strip() for s in sents if s and s.strip()]

    def chunk(self, doc: Document) -> list[Chunk]:
        sents = self._sentences(doc.text)
        if len(sents) <= 1:
            return [Chunk(text=doc.text.strip(), metadata=dict(doc.metadata))]

        vecs = np.array(self.embedder.embed_documents(sents))
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        dists = 1 - (vecs[:-1] * vecs[1:]).sum(axis=1)   # 相鄰句的 cosine 距離
        threshold = np.percentile(dists, self.percentile)

        groups: list[list[str]] = []
        buf = [sents[0]]
        for i, d in enumerate(dists):
            nxt = sents[i + 1]
            too_big = sum(map(len, buf)) + len(nxt) > self.max_chunk_size
            if d > threshold or too_big:
                groups.append(buf)
                buf = []
            buf.append(nxt)
        if buf:
            groups.append(buf)

        return [Chunk(text=" ".join(g), metadata=dict(doc.metadata)) for g in groups]
