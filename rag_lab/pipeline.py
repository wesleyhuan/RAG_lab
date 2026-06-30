from functools import cached_property

from rag_lab.chunking import CHUNKERS
from rag_lab.config import RAGConfig
from rag_lab.documents import Document, Chunk
from rag_lab.embedding import EMBEDDERS
from rag_lab.llm import LLMS
from rag_lab.retrieval import RETRIEVERS
from rag_lab.vectorstore import VECTOR_STORES

RAG_PROMPT = """你是一個根據文件回答問題的助手。請「只」根據以下參考內容回答；
若參考內容不足以回答，請直接說不知道，不要編造。

[參考內容]
{context}

[問題]
{question}"""


class RAGPipeline:
    """把 RAGConfig（字串描述）組裝成可運作的物件圖。
    元件（embedder/store/llm/reranker）採延遲建立：用到才載，載一次（cached_property），
    所以「選齊策略、建構 pipeline」本身不載任何權重。
    依賴注入發生在 chunk()：semantic 拿到 embedder、contextual / HyDE 拿到 llm。"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] | None = None
        self.retriever = None
        self.final_k = config.retriever_params.get("top_k", 5)

    @cached_property
    def llm(self):
        return LLMS.create(self.config.llm_backend, **self.config.llm_params)

    @cached_property
    def embedder(self):
        return EMBEDDERS.create(self.config.embedder, **self.config.embedder_params)

    @cached_property
    def store(self):
        return VECTOR_STORES.create(self.config.vector_store)

    @cached_property
    def reranker(self):
        if not self.config.use_reranker:
            return None
        # 延遲 import：CrossEncoderReranker 會載 sentence_transformers（重）
        from rag_lab.rerank import CrossEncoderReranker
        return CrossEncoderReranker(self.config.reranker_model)

    def chunk(self, docs: list[Document]) -> list[Chunk]:
        # 只注入該 chunker 宣告需要的依賴（recursive NEEDS=() -> 不碰 embedder/llm）
        deps = {name: getattr(self, name)
                for name in CHUNKERS.get(self.config.chunker).NEEDS}
        chunker = CHUNKERS.create(self.config.chunker, **deps, **self.config.chunker_params)
        self.chunks = chunker.chunk_all(docs)
        return self.chunks

    def embed(self) -> list[list[float]]:
        self.vectors = self.embedder.embed_documents([c.text for c in self.chunks])
        return self.vectors

    def index(self) -> None:
        self.store.add(self.chunks, self.vectors)

    def build_retriever(self):
        # 有 reranker 時讓 retriever 多撈 4 倍候選，再由 cross-encoder 精排回 final_k
        fetch_k = self.final_k * 4 if self.reranker else self.final_k
        # llm/embedder 在此一併傳入：建構 LLM 物件很便宜（Ollama 要到 generate 才載模型），
        # 真正吃資源的權重已在前面 embed 步驟載完。
        params = {**self.config.retriever_params, "top_k": fetch_k}
        self.retriever = RETRIEVERS.create(
            self.config.retriever, embedder=self.embedder, store=self.store,
            chunks=self.chunks, llm=self.llm, **params)
        return self.retriever

    def build_index(self, docs: list[Document], progress=None) -> int:
        def log(msg):
            if progress:
                progress(msg)

        log(f"切割文件中（{self.config.chunker}）…")
        self.chunk(docs)
        log(f"共 {len(self.chunks)} 個 chunks，向量化中（{self.config.embedder}）…")
        self.embed()
        self.index()
        log(f"建立 retriever（{self.config.retriever}）…")
        self.build_retriever()
        return len(self.chunks)

    def retrieve(self, query: str):
        results = self.retriever.retrieve(query)
        if self.reranker:
            results = self.reranker.rerank(query, results, top_k=self.final_k)
        return results

    def answer(self, query: str, stream: bool = False):
        results = self.retrieve(query)
        context = "\n\n---\n\n".join(c.text for c, _ in results)
        prompt = RAG_PROMPT.format(context=context, question=query)
        if stream:
            return self.llm.stream(prompt), results
        return self.llm.generate(prompt), results
