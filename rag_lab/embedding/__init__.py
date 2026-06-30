from .base import EMBEDDERS, BaseEmbedder  # noqa: F401

# 延遲註冊：local 會 import sentence_transformers（重），到 create() 才載入
EMBEDDERS.lazy(
    minilm="rag_lab.embedding.local",
    bge="rag_lab.embedding.local",
    openai="rag_lab.embedding.openai_embed",
)
