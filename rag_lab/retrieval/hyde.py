from .base import BaseRetriever, RETRIEVERS

HYDE_PROMPT = """請針對以下問題，寫一段「可能出現在參考文件中」的假設性回答。
直接寫內容，不要任何前言或免責聲明，一個段落即可。
問題：{q}"""


@RETRIEVERS.register("hyde")
class HyDERetriever(BaseRetriever):
    """HyDE（Hypothetical Document Embeddings）：
    「問題」和「答案段落」在向量空間的距離常常不近（問句 vs 陳述句），
    所以先讓 LLM 寫一段假設性答案，用「假答案的向量」去找「真答案」——
    答案跟答案之間語意更接近。代價：每次查詢多一次 LLM 呼叫。"""

    def retrieve(self, query):
        hypothetical = self.llm.generate(HYDE_PROMPT.format(q=query))
        vec = self.embedder.embed_query(hypothetical)
        return self.store.search(vec, self.top_k)
