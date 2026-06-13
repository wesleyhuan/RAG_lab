import numpy as np

from .base import BaseRetriever, RETRIEVERS


@RETRIEVERS.register("mmr")
class MMRRetriever(BaseRetriever):
    """MMR（Maximal Marginal Relevance）：
    先撈 fetch_k 個候選，再貪婪挑選——每次挑「跟問題相關、但跟已選的不重複」的：
        MMR(i) = λ * sim(query, i) - (1-λ) * max_j sim(i, 已選 j)
    λ 越大越重相關性，越小越重多樣性。適合文件內容重複度高的情境。"""

    def __init__(self, lambda_mult: float = 0.6, fetch_k: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.lambda_mult = lambda_mult
        self.fetch_k = max(fetch_k, self.top_k * 2)

    def retrieve(self, query):
        qv = np.array(self.embedder.embed_query(query))
        candidates = self.store.search(qv.tolist(), self.fetch_k)
        if not candidates:
            return []

        cand_chunks = [c for c, _ in candidates]
        cv = np.array(self.embedder.embed_documents([c.text for c in cand_chunks]))
        cv = cv / (np.linalg.norm(cv, axis=1, keepdims=True) + 1e-10)
        qv = qv / (np.linalg.norm(qv) + 1e-10)
        sim_q = cv @ qv

        selected: list[int] = []
        remaining = list(range(len(cand_chunks)))
        while remaining and len(selected) < self.top_k:
            if not selected:
                best = max(remaining, key=lambda i: sim_q[i])
            else:
                def mmr_score(i):
                    max_sim = max(float(cv[i] @ cv[j]) for j in selected)
                    return self.lambda_mult * sim_q[i] - (1 - self.lambda_mult) * max_sim
                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)

        return [(cand_chunks[i], float(sim_q[i])) for i in selected]
