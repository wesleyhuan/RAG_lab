from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder, EMBEDDERS


class _STEmbedder(BaseEmbedder):
    """sentence-transformers 共用邏輯，子類只要指定模型名稱"""
    MODEL_NAME = ""
    QUERY_PREFIX = ""

    def __init__(self, model_name: str = "", **_):
        self.model = SentenceTransformer(model_name or self.MODEL_NAME)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [self.QUERY_PREFIX + text], normalize_embeddings=True
        )[0].tolist()


@EMBEDDERS.register("minilm")
class MiniLMEmbedder(_STEmbedder):
    """輕量英文模型，384 維，速度快；中文效果普通"""
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@EMBEDDERS.register("bge")
class BGEEmbedder(_STEmbedder):
    """BAAI BGE 中文模型；官方建議「查詢」要加指令 prefix，文件不用——
    這就是 embed_query / embed_documents 要分開的原因"""
    MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 英文資料可換 BAAI/bge-small-en-v1.5
    QUERY_PREFIX = "為這個句子生成表示以用於檢索相關文章："
