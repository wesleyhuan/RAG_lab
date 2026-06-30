from .base import CHUNKERS, BaseChunker  # noqa: F401

# 延遲註冊：semantic 會 import numpy，到 create() 才載入
CHUNKERS.lazy(
    recursive="rag_lab.chunking.recursive",
    semantic="rag_lab.chunking.semantic",
    contextual="rag_lab.chunking.contextual",
)
