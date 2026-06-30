"""Registry.get() 回傳已註冊的類別；每個 chunker 用 NEEDS 宣告它需要哪些依賴
（pipeline.chunk() 據此只注入該 chunker 真正要的東西，避免 recursive 白載 embedder）。
無 pytest，直接 `PYTHONPATH=. python tests/test_chunker_needs.py` 執行。
"""


def test_registry_get_returns_class():
    from rag_lab.chunking import CHUNKERS
    cls = CHUNKERS.get("recursive")
    assert isinstance(cls, type), f"get() 應回傳類別，得到 {cls!r}"


def test_chunker_needs():
    from rag_lab.chunking import CHUNKERS
    assert CHUNKERS.get("recursive").NEEDS == ()
    assert CHUNKERS.get("semantic").NEEDS == ("embedder",)
    assert CHUNKERS.get("contextual").NEEDS == ("llm",)


if __name__ == "__main__":
    test_registry_get_returns_class()
    test_chunker_needs()
    print("PASS: Registry.get 與 chunker NEEDS 正確")
