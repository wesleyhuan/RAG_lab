class Registry:
    """註冊表：用字串 key 對應到類別，是整個系統「可替換策略」的核心。

    用法：
        CHUNKERS = Registry("chunker")

        @CHUNKERS.register("recursive")
        class RecursiveChunker: ...

        chunker = CHUNKERS.create("recursive", chunk_size=500)
    """

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, type] = {}

    def register(self, key: str):
        def decorator(cls):
            self._items[key] = cls
            return cls
        return decorator

    def create(self, key: str, **kwargs):
        if key not in self._items:
            raise KeyError(f"{self.name} 沒有 '{key}'，可用：{list(self._items)}")
        return self._items[key](**kwargs)

    def keys(self) -> list[str]:
        return sorted(self._items)
