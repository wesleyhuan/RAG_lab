from .base import BaseRetriever, RETRIEVERS


@RETRIEVERS.register("dense")
class DenseRetriever(BaseRetriever):
    """最基本的密集向量檢索，當作 baseline 對照組"""

    def retrieve(self, query):
        return self.store.search(self.embedder.embed_query(query), self.top_k)
