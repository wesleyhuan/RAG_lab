from .base import RETRIEVERS, BaseRetriever  # noqa: F401

# 延遲註冊：hybrid 會 import jieba / rank_bm25，到 create() 才載入
RETRIEVERS.lazy(
    dense="rag_lab.retrieval.dense",
    hybrid="rag_lab.retrieval.hybrid",
    hyde="rag_lab.retrieval.hyde",
    mmr="rag_lab.retrieval.mmr",
)
