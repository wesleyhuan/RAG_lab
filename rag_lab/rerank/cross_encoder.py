import streamlit as st
from sentence_transformers import CrossEncoder

from rag_lab.torch_runtime import limit_torch_threads


@st.cache_resource(show_spinner=False)
def _load_model(model: str) -> CrossEncoder:
    """以模型名稱為 key 快取 cross-encoder 權重,重建時重用。"""
    limit_torch_threads()          # 載模型前先限制執行緒，避免 predict 吃滿 CPU
    return CrossEncoder(model)


class CrossEncoderReranker:
    """兩階段檢索的第二階段。
    Bi-encoder（embedding）：問題、文件「分開」編碼 -> 快，可預先建索引，但較不準。
    Cross-encoder：問題+文件「一起」進模型打分 -> 準，但慢，無法預先計算。
    所以策略是：retriever 先粗篩多一點候選，cross-encoder 只重排這一小批。"""

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        self.model = _load_model(model)

    def rerank(self, query, results, top_k: int = 5):
        if not results:
            return []
        scores = self.model.predict([(query, c.text) for c, _ in results])
        ranked = sorted(zip(results, scores), key=lambda x: -x[1])
        return [(c, float(s)) for (c, _), s in ranked[:top_k]]
