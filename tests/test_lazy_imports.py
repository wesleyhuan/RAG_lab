"""進入「建立索引」頁時，只 import 頁面需要的 registry / pipeline，
不該連帶把 torch / sentence_transformers / faiss 等重函式庫一起載入
（那是 34s loading、整頁卡住的元兇）。重函式庫應延遲到真正建立 pipeline 時才載。

無 pytest，直接 `python tests/test_lazy_imports.py` 執行；exit code 0 = 通過。
"""
import sys

HEAVY = ["torch", "sentence_transformers", "transformers",
         "faiss", "chromadb", "qdrant_client"]


def test_page_imports_stay_light():
    # 模擬 pages/2_🔧_建立索引.py 的 import（state 會拉 streamlit，與本測試無關故略過）
    from rag_lab.chunking import CHUNKERS
    from rag_lab.config import RAGConfig          # noqa: F401
    from rag_lab.embedding import EMBEDDERS
    from rag_lab.llm import LLMS
    from rag_lab.pipeline import RAGPipeline      # noqa: F401
    from rag_lab.retrieval import RETRIEVERS
    from rag_lab.vectorstore import VECTOR_STORES

    leaked = [m for m in HEAVY if m in sys.modules]
    assert not leaked, f"進頁就載入了重函式庫：{leaked}"

    # 延遲化不能犧牲功能：選單該列的策略名稱仍要在
    assert "minilm" in EMBEDDERS.keys()
    assert "faiss" in VECTOR_STORES.keys()
    assert "recursive" in CHUNKERS.keys()
    assert "dense" in RETRIEVERS.keys()
    assert "ollama" in LLMS.keys()


if __name__ == "__main__":
    test_page_imports_stay_light()
    print("PASS: 頁面 import 維持輕量，重函式庫未被載入")
