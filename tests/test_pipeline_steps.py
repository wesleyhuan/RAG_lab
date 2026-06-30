"""RAGPipeline 的元件改為延遲建立（用到才載），build_index 由四個分步方法組成。
驗證：(1) 建構不載入任何元件；(2) recursive 的 chunk() 不會觸發 embedder；
(3) build_index 依序呼叫四個分步方法。
無 pytest，直接 `PYTHONPATH=. python tests/test_pipeline_steps.py` 執行。
"""
from rag_lab.config import RAGConfig
from rag_lab.documents import Document
from rag_lab.pipeline import RAGPipeline


def test_construction_loads_nothing():
    pipe = RAGPipeline(RAGConfig())
    for name in ("embedder", "store", "llm", "reranker"):
        assert name not in pipe.__dict__, f"建構就建立了 {name}（cached_property 不該被觸發）"


def test_recursive_chunk_does_not_load_embedder():
    pipe = RAGPipeline(RAGConfig(chunker="recursive"))
    doc = Document(text="第一段。第二段！第三段？" * 50)
    chunks = pipe.chunk([doc])
    assert len(chunks) > 0, "recursive 應切出至少一個 chunk"
    assert "embedder" not in pipe.__dict__, "recursive 切割不該觸發 embedder"
    assert pipe.chunks == chunks, "chunk() 應同時更新 self.chunks（後續 embed() 會讀它）"


def test_build_index_calls_steps_in_order():
    pipe = RAGPipeline(RAGConfig())
    calls = []
    pipe.chunk = lambda docs: calls.append("chunk") or []
    pipe.embed = lambda: calls.append("embed")
    pipe.index = lambda: calls.append("index")
    pipe.build_retriever = lambda: calls.append("retriever")
    pipe.build_index([])
    assert calls == ["chunk", "embed", "index", "retriever"], f"順序錯誤：{calls}"


if __name__ == "__main__":
    test_construction_loads_nothing()
    test_recursive_chunk_does_not_load_embedder()
    test_build_index_calls_steps_in_order()
    print("PASS: pipeline 延遲建立 + 分步方法正確")
