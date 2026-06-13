from dataclasses import dataclass, field


@dataclass
class Document:
    """一份原始文件（一個 PDF、一個網頁…）"""
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """切割後的片段，id 是全域索引（BM25 / 向量庫共用）"""
    text: str
    metadata: dict = field(default_factory=dict)
    id: int = -1
