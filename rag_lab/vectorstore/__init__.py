from .base import VECTOR_STORES, BaseVectorStore  # noqa: F401

# 延遲註冊：faiss / chromadb / qdrant_client 都是重 import，到 create() 才載入
VECTOR_STORES.lazy(
    faiss="rag_lab.vectorstore.faiss_store",
    chroma="rag_lab.vectorstore.chroma_store",
    qdrant="rag_lab.vectorstore.qdrant_store",
)
