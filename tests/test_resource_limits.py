"""驗證兩個資源保護機制：
  4) torch 的 CPU 執行緒被限制——避免 embedding/rerank 把所有核心吃滿，
     導致 Streamlit server 線程搶不到 CPU、websocket 斷線（看起來像「server 停了」）。
  5) embedding 分批編碼——batch_size 控制單次進模型的量，壓低記憶體峰值。
無 pytest，直接 `python tests/test_resource_limits.py` 執行；exit code 0 = 通過。
"""


def test_torch_threads_respects_env():
    import os

    os.environ["RAG_LAB_TORCH_THREADS"] = "1"
    import rag_lab.torch_runtime as rt
    rt._applied = False                       # 重置，確保這次真的重新套用
    rt.limit_torch_threads()

    import torch
    assert torch.get_num_threads() == 1, \
        f"應限制為 1 顆核，實際 {torch.get_num_threads()}"


def test_embed_documents_uses_batch_size():
    import rag_lab.embedding.local as local

    class _Arr(list):
        def tolist(self):
            return [list(row) for row in self]

    class _FakeModel:
        def __init__(self, name):
            self.last_batch_size = None

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, batch_size=32, **_):
            self.last_batch_size = batch_size
            return _Arr([[0.0, 0.0, 0.0] for _ in texts])

    local.SentenceTransformer = _FakeModel
    try:
        local._load_model.clear()             # 清掉 st.cache_resource 既有快取
    except Exception:
        pass

    emb = local.MiniLMEmbedder(batch_size=8)
    emb.embed_documents(["a", "b", "c"])
    assert emb.model.last_batch_size == 8, \
        f"encode 應收到 batch_size=8，實際 {emb.model.last_batch_size}"


if __name__ == "__main__":
    test_torch_threads_respects_env()
    test_embed_documents_uses_batch_size()
    print("PASS: 執行緒限制與 embedding 分批生效")
