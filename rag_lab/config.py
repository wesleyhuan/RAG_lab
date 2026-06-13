from dataclasses import dataclass, field, asdict


@dataclass
class RAGConfig:
    """一組 RAG 架構配置。只存「字串 + 參數」，不存物件，
    所以可以序列化、比較、存檔——組裝交給 RAGPipeline。"""
    name: str = "default"

    chunker: str = "recursive"            # recursive / semantic / contextual
    chunker_params: dict = field(default_factory=dict)

    embedder: str = "minilm"              # minilm / bge / openai
    embedder_params: dict = field(default_factory=dict)

    vector_store: str = "faiss"           # faiss / chroma / qdrant

    retriever: str = "dense"              # dense / hybrid / hyde / mmr
    retriever_params: dict = field(default_factory=dict)  # 例：{"top_k": 5}

    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"

    llm_backend: str = "ollama"           # ollama / openai
    llm_params: dict = field(default_factory=dict)        # 例：{"model": "qwen2.5:7b"}

    def to_dict(self) -> dict:
        return asdict(self)
