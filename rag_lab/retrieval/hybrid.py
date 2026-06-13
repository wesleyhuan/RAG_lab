import jieba
from rank_bm25 import BM25Okapi

from .base import BaseRetriever, RETRIEVERS


@RETRIEVERS.register("hybrid")
class HybridRetriever(BaseRetriever):
    """混合檢索：BM25（關鍵字精準比對）+ 向量（語意相似），
    用 RRF（Reciprocal Rank Fusion）合併兩邊的「排名」而不是分數——
    因為 BM25 分數和 cosine 分數的量綱不同，直接加權不公平。
    中文沒有空格分詞，所以用 jieba 斷詞餵給 BM25。"""

    def __init__(self, rrf_k: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([list(jieba.cut(c.text)) for c in self.chunks])

    def retrieve(self, query):
        n = max(self.top_k * 3, 10)

        # 1) 向量這一路
        dense = self.store.search(self.embedder.embed_query(query), n)

        # 2) BM25 這一路
        scores = self.bm25.get_scores(list(jieba.cut(query)))
        sparse_ids = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]

        # 3) RRF 融合：score = sum( 1 / (k + rank) )
        fused: dict[int, float] = {}
        for rank, (chunk, _) in enumerate(dense):
            fused[chunk.id] = fused.get(chunk.id, 0) + 1 / (self.rrf_k + rank + 1)
        for rank, cid in enumerate(sparse_ids):
            chunk_id = self.chunks[cid].id
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (self.rrf_k + rank + 1)

        by_id = {c.id: c for c in self.chunks}
        top = sorted(fused.items(), key=lambda x: -x[1])[: self.top_k]
        return [(by_id[cid], score) for cid, score in top]
