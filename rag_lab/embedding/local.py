import logging

import streamlit as st
from sentence_transformers import SentenceTransformer

from rag_lab.torch_runtime import limit_torch_threads
from .base import BaseEmbedder, EMBEDDERS

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _load_model(model_name: str) -> SentenceTransformer:
    """以模型名稱為 key 快取已載入的權重。重建索引(換配置但同一個模型)時
    直接重用,不必每次把 ~80MB~1GB 權重重新讀進記憶體。"""
    limit_torch_threads()          # 載模型前先限制執行緒，避免 encode 吃滿 CPU
    return SentenceTransformer(model_name)


class _STEmbedder(BaseEmbedder):
    """sentence-transformers 共用邏輯，子類只要指定模型名稱"""
    MODEL_NAME = ""
    QUERY_PREFIX = ""

    def __init__(self, model_name: str = "", batch_size: int = 16, **_):
        self.model = _load_model(model_name or self.MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size   # 分批編碼，壓低記憶體峰值（預設小於官方 32）

    def embed_documents(self, texts):
        logger.debug("embedding %d 段文字，batch_size=%d", len(texts), self.batch_size)
        return self.model.encode(
            texts, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [self.QUERY_PREFIX + text], normalize_embeddings=True
        )[0].tolist()


@EMBEDDERS.register("minilm")
class MiniLMEmbedder(_STEmbedder):
    """輕量英文模型，384 維，速度快；中文效果普通"""
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@EMBEDDERS.register("bge")
class BGEEmbedder(_STEmbedder):
    """BAAI BGE 中文模型；官方建議「查詢」要加指令 prefix，文件不用——
    這就是 embed_query / embed_documents 要分開的原因"""
    MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 英文資料可換 BAAI/bge-small-en-v1.5
    QUERY_PREFIX = "為這個句子生成表示以用於檢索相關文章："
