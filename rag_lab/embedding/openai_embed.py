from openai import OpenAI

from .base import BaseEmbedder, EMBEDDERS


@EMBEDDERS.register("openai")
class OpenAIEmbedder(BaseEmbedder):
    """需要環境變數 OPENAI_API_KEY"""

    def __init__(self, model: str = "text-embedding-3-small", **_):
        self.client = OpenAI()
        self.model = model
        self.dim = 1536

    def embed_documents(self, texts):
        out = []
        for i in range(0, len(texts), 100):     # 分批避免單次 request 過大
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + 100])
            out.extend(d.embedding for d in resp.data)
        return out
