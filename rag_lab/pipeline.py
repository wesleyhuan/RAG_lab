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
    依賴注入發生在這裡：semantic chunker 拿到 embedder、
    contextual chunker / HyDE 拿到 llm。"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = LLMS.create(config.llm_backend, **config.llm_params)
        self.embedder = EMBEDDERS.create(config.embedder, **config.embedder_params)
        self.store = VECTOR_STORES.create(config.vector_store)
        self.reranker = self._make_reranker(config)
        self.chunks: list[Chunk] = []
        self.retriever = None
        self.final_k = config.retriever_params.get("top_k", 5)

    @staticmethod
    def _make_reranker(config: RAGConfig):
        if not config.use_reranker:
            return None
        # 延遲 import：CrossEncoderReranker 會載 sentence_transformers（重），
        # 只有真的開 reranker 才付這個成本
        from rag_lab.rerank import CrossEncoderReranker
        return CrossEncoderReranker(config.reranker_model)

    def build_index(self, docs: list[Document], progress=None) -> int:
        def log(msg):
            if progress:
                progress(msg)

        log(f"切割文件中（{self.config.chunker}）…")
        chunker = CHUNKERS.create(self.config.chunker,
                                  embedder=self.embedder, llm=self.llm,
                                  **self.config.chunker_params)
        self.chunks = chunker.chunk_all(docs)

        log(f"共 {len(self.chunks)} 個 chunks，向量化中（{self.config.embedder}）…")
        vectors = self.embedder.embed_documents([c.text for c in self.chunks])
        self.store.add(self.chunks, vectors)

        log(f"建立 retriever（{self.config.retriever}）…")
        # 有 reranker 時讓 retriever 多撈 4 倍候選，再由 cross-encoder 精排回 final_k
        fetch_k = self.final_k * 4 if self.reranker else self.final_k
        params = {**self.config.retriever_params, "top_k": fetch_k}
        self.retriever = RETRIEVERS.create(
            self.config.retriever,
            embedder=self.embedder, store=self.store,
            chunks=self.chunks, llm=self.llm, **params)
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
