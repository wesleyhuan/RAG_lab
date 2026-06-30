import importlib


class Registry:
    """註冊表：用字串 key 對應到類別，是整個系統「可替換策略」的核心。

    用法：
        CHUNKERS = Registry("chunker")

        @CHUNKERS.register("recursive")
        class RecursiveChunker: ...

        chunker = CHUNKERS.create("recursive", chunk_size=500)

    延遲載入：實作模組常會 import torch / faiss 等重函式庫。若 __init__ 一律
    eager import 它們，光是「列出選單」就得載入整套 ML stack（實測 ~34s）。
    所以改成在 __init__ 用 lazy() 宣告「key -> 實作模組路徑」，真正 create()
    時才 import 該模組（觸發其 @register），把成本延到建立 pipeline 那一刻。
    """

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, type] = {}       # @register 觸發後填入：key -> class
        self._modules: dict[str, str] = {}      # 延遲宣告：key -> 實作模組路徑

    def register(self, key: str):
        def decorator(cls):
            self._items[key] = cls
            return cls
        return decorator

    def lazy(self, **key_to_module: str):
        """宣告 key 所屬的實作模組，import 延遲到 create()。
        例：EMBEDDERS.lazy(minilm="rag_lab.embedding.local")"""
        self._modules.update(key_to_module)

    def _resolve(self, key: str) -> type:
        if key not in self._items and key in self._modules:
            importlib.import_module(self._modules[key])   # 觸發該模組的 @register
        if key not in self._items:
            raise KeyError(f"{self.name} 沒有 '{key}'，可用：{self.keys()}")
        return self._items[key]

    def create(self, key: str, **kwargs):
        return self._resolve(key)(**kwargs)

    def keys(self) -> list[str]:
        return sorted({*self._items, *self._modules})
