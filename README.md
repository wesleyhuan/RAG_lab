# 🧪 RAG Lab — 可組合的 RAG 實驗平台

自由組合 **Chunking × Embedding × Vector Store × Retrieval × Reranking**，
上傳自己的文件或網頁，用 **RAGAS** 評估各種架構組合，選出最好的那組來對話。

## 架構總覽

```
                    RAGConfig（純字串+參數，可序列化比較）
                              │
                        RAGPipeline（組裝器，依賴注入發生在這）
                              │
   ┌──────────┬──────────┬───┴──────┬───────────┬──────────┐
 Chunking   Embedding  VectorStore  Retrieval   Rerank      LLM
 recursive   minilm      faiss       dense      cross-     ollama
 semantic    bge         chroma      hybrid     encoder    openai
 contextual  openai      qdrant      hyde
                                     mmr
```

核心機制只有兩個，都在 `rag_lab/registry.py` 和 `rag_lab/pipeline.py`：

1. **Registry pattern**：每個策略類別用 `@CHUNKERS.register("recursive")` 這種
   decorator 註冊進字典，UI 的下拉選單直接讀 `CHUNKERS.keys()`，
   新增策略 = 加一個檔案，UI 自動出現。
2. **依賴注入**：semantic chunker 需要 embedder、contextual/HyDE 需要 LLM，
   全部由 `RAGPipeline.__init__` 統一建立後注入，元件之間不互相 import。

## 快速開始

```bash
git clone https://github.com/wesleyhuan/RAG_lab.git
cd RAG_lab

python -m venv venv && venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 本地 LLM（擇一）：
ollama pull qwen2.5:7b
ollama pull nomic-embed-text    # RAGAS 用 ollama 當裁判時需要

# 或 OpenAI：複製 .env.example -> 設定 OPENAI_API_KEY 環境變數

streamlit run app.py
```

## 操作流程

啟動後依左側頁面順序操作（`pages/` 目錄）：

| 步驟 | 頁面 | 做什麼 |
|---|---|---|
| 1 | **📁 資料來源** | 上傳 PDF / TXT / MD，或輸入網址抓取網頁 |
| 2 | **🔧 建立索引** | 組合 chunking × embedding × vector store × retrieval × rerank，可建立多組配置 |
| 3 | **📊 評估比較** | 輸入測試問題，用 RAGAS 對所有配置跑分、並排比較 |
| 4 | **💬 對話** | 選一個（評估後勝出的）配置開始問答，可展開看檢索到的片段 |

## 學習路線（建議閱讀順序）

| 順序 | 檔案 | 學什麼 |
|---|---|---|
| 1 | `rag_lab/registry.py` | 整個系統可替換的核心：30 行的 Registry |
| 2 | `rag_lab/documents.py`, `config.py` | 資料模型與配置的分離 |
| 3 | `rag_lab/chunking/recursive.py` | 從零實作遞迴切割（不靠 LangChain） |
| 4 | `rag_lab/chunking/semantic.py` | 用相鄰句向量距離找主題斷點 |
| 5 | `rag_lab/chunking/contextual.py` | Anthropic Contextual Retrieval 的做法 |
| 6 | `rag_lab/embedding/local.py` | 為什麼 BGE 的 query 要加 prefix |
| 7 | `rag_lab/vectorstore/*` | 同一個介面下三種向量庫的差異 |
| 8 | `rag_lab/retrieval/hybrid.py` | BM25 + 向量，為什麼用 RRF 融合排名而非分數 |
| 9 | `rag_lab/retrieval/hyde.py`, `mmr.py` | HyDE 的直覺、MMR 的貪婪選擇公式 |
| 10 | `rag_lab/rerank/cross_encoder.py` | bi-encoder vs cross-encoder 的取捨 |
| 11 | `rag_lab/pipeline.py` | 兩階段檢索：粗篩 4×k -> 精排回 k |
| 12 | `rag_lab/evaluation/ragas_eval.py` | 四個 RAGAS 指標各自在量什麼 |

## 注意事項

- **contextual chunking** 每個 chunk 都呼叫一次 LLM，文件大時建索引很慢（先拿小文件試）。
- **RAGAS 裁判**：用本地小模型當裁判分數會偏不穩，預算允許時建議裁判用 OpenAI。
- **Ollama context**：`OllamaLLM` 已強制 `num_ctx=8192`，避免 RAG prompt 被截斷。
- 所有索引都在記憶體（重開 Streamlit 就消失）；要持久化見 `chroma_store.py` / `qdrant_store.py` 內註解。
- 評估時間 ≈ 配置數 × 題數 ×（1 次回答 + 數次裁判呼叫），先用 3–5 題小規模試。
- **相依性**：`ragas` 仍會 import 已被 `langchain-community` 0.4（sunset 版）移除的 `chat_models.vertexai`，
  故 `requirements.txt` 將 `langchain-community` 鎖在 `>=0.3,<0.4`。若評估頁出現
  `No module named 'langchain_community.chat_models.vertexai'`，請改裝 0.3.x。

## 擴充示範：加一個新的 retriever

```python
# rag_lab/retrieval/my_strategy.py
from .base import BaseRetriever, RETRIEVERS

@RETRIEVERS.register("my_strategy")
class MyRetriever(BaseRetriever):
    def retrieve(self, query):
        ...

# 然後在 rag_lab/retrieval/__init__.py 加一行 import，UI 下拉選單就會出現
```
