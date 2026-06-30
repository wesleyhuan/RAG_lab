"""同一個模型名稱只該載入一次，重建索引時要重用已載入的模型物件,
不要每次都重新把 ~80MB~1GB 的權重讀進記憶體。

用假模型取代真正的 SentenceTransformer / CrossEncoder(避免下載權重),
驗證「兩個用相同模型名的元件共用同一個底層 model 物件」。
無 pytest,直接 `python tests/test_model_caching.py` 執行。
"""


class _FakeModel:
    instances = 0

    def __init__(self, name):
        type(self).instances += 1
        self.name = name

    def get_sentence_embedding_dimension(self):
        return 3


def test_sentence_transformer_loaded_once_per_name():
    import rag_lab.embedding.local as local
    local.SentenceTransformer = _FakeModel
    _FakeModel.instances = 0

    e1 = local.MiniLMEmbedder()
    e2 = local.MiniLMEmbedder()

    assert _FakeModel.instances == 1, f"模型被載入 {_FakeModel.instances} 次,應只 1 次"
    assert e1.model is e2.model, "相同模型名的兩個 embedder 應共用同一個 model 物件"


def test_cross_encoder_loaded_once_per_name():
    import rag_lab.rerank.cross_encoder as ce
    ce.CrossEncoder = _FakeModel
    _FakeModel.instances = 0

    r1 = ce.CrossEncoderReranker("dummy-rerank")
    r2 = ce.CrossEncoderReranker("dummy-rerank")

    assert _FakeModel.instances == 1, f"reranker 被載入 {_FakeModel.instances} 次,應只 1 次"
    assert r1.model is r2.model, "相同模型名的兩個 reranker 應共用同一個 model 物件"


if __name__ == "__main__":
    test_sentence_transformer_loaded_once_per_name()
    test_cross_encoder_loaded_once_per_name()
    print("PASS: 相同模型名只載入一次,重建索引可重用")
