from .base import BaseChunker, CHUNKERS
from .recursive import RecursiveChunker
from rag_lab.documents import Document, Chunk

CONTEXT_PROMPT = """<document>
{doc}
</document>
以下是上面文件中的一個段落：
<chunk>
{chunk}
</chunk>
請用一到兩句話，說明這個段落在整份文件中的位置與主題（用來改善搜尋檢索）。只輸出說明文字本身。"""


@CHUNKERS.register("contextual")
class ContextualChunker(BaseChunker):
    """Contextual Retrieval（Anthropic 提出的做法）：
    先用 recursive 切，再讓 LLM 看「整份文件 + 這個 chunk」，
    為每個 chunk 生成一段脈絡說明並接在前面，讓孤立的片段也帶有全文資訊。
    優點：檢索準確度通常明顯提升。缺點：建索引慢、要花 LLM token。"""

    NEEDS = ("llm",)

    def __init__(self, llm=None, chunk_size: int = 500, overlap: int = 50,
                 max_doc_chars: int = 8000, **_):
        if llm is None:
            raise ValueError("contextual chunker 需要注入 llm")
        self.llm = llm
        self.inner = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
        self.max_doc_chars = max_doc_chars   # 控制 prompt 長度（本地模型 context 有限）

    def chunk(self, doc: Document) -> list[Chunk]:
        base_chunks = self.inner.chunk(doc)
        doc_text = doc.text[: self.max_doc_chars]
        out = []
        for c in base_chunks:
            ctx = self.llm.generate(
                CONTEXT_PROMPT.format(doc=doc_text, chunk=c.text)
            ).strip()
            out.append(Chunk(text=f"{ctx}\n{c.text}",
                             metadata={**c.metadata, "context": ctx}))
        return out
