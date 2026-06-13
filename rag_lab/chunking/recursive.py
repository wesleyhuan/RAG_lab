from .base import BaseChunker, CHUNKERS
from rag_lab.documents import Document, Chunk


@CHUNKERS.register("recursive")
class RecursiveChunker(BaseChunker):
    """規則式遞迴切割（自己實作，不用 LangChain）：
    優先用大的分隔符（段落）切，片段太大就遞迴用更細的分隔符（句子、逗號…）。
    優點：快、零成本。缺點：可能在語意中間切斷。"""

    SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "，", " ", ""]

    def __init__(self, chunk_size: int = 500, overlap: int = 50, **_):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split(self, text: str, sep_idx: int = 0) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        sep = self.SEPARATORS[sep_idx] if sep_idx < len(self.SEPARATORS) else ""
        if sep == "":
            # 已無分隔符可用 -> 硬切
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) <= 1:
            return self._split(text, sep_idx + 1)

        out: list[str] = []
        buf = ""
        for p in parts:
            piece = p + sep
            if len(piece) > self.chunk_size:
                # 單一片段就超標 -> 先收掉 buffer，再遞迴切這個片段
                if buf.strip():
                    out.append(buf)
                out.extend(self._split(piece, sep_idx + 1))
                buf = ""
                continue
            if len(buf) + len(piece) > self.chunk_size and buf:
                out.append(buf)
                buf = buf[-self.overlap:] if self.overlap else ""  # 保留尾巴當 overlap
            buf += piece
        if buf.strip():
            out.append(buf)
        return out

    def chunk(self, doc: Document) -> list[Chunk]:
        return [Chunk(text=t.strip(), metadata=dict(doc.metadata))
                for t in self._split(doc.text) if t.strip()]
