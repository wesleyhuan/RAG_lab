# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

可組合的 RAG 實驗平台：自由組合 Chunking × Embedding × Vector Store × Retrieval × Rerank × LLM，用 RAGAS 評估後挑勝出配置對話。完整使用說明見 `README.md`（中文）。

## 指令

```bash
# 環境（Windows）
venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py          # 啟動 UI，從 app.py 進入，pages/ 依檔名數字排序

# 測試：沒有 pytest，每個檔案直接執行，exit code 0 = 通過
# 需設 PYTHONPATH=. 讓 rag_lab 可被 import（否則 sys.path 只有 tests/，會 ModuleNotFoundError）
PYTHONPATH=. python tests/test_lazy_imports.py        # PowerShell: $env:PYTHONPATH="."; python tests/...
PYTHONPATH=. python tests/test_model_caching.py
PYTHONPATH=. python tests/test_resource_limits.py
```

本地 LLM 用 Ollama（`ollama pull qwen2.5:7b`、`nomic-embed-text`）；或設 `OPENAI_API_KEY`（複製 `.env.example`）。

## 核心機制

整個系統的可替換性建立在兩個檔案上，動任何策略前先讀懂它們：

- **`rag_lab/registry.py`** — `Registry` 用字串 key 對應類別。關鍵是**延遲載入**：策略實作常 import torch/faiss/sentence-transformers 等重函式庫（eager import 全載 ~34s）。所以每個類別的 `__init__.py` 只用 `REGISTRY.lazy(key="module.path")` 宣告「key → 模組路徑」，真正 `create()` 時才 import 該模組以觸發其 `@register`。`keys()` 同時看 `_items` 和 `_modules`，所以選單能列出尚未載入的策略。

- **`rag_lab/pipeline.py`** — `RAGPipeline` 把 `RAGConfig`（純字串+參數，可序列化/比較）組裝成物件圖。**依賴注入只發生在這裡**：semantic chunker 拿到 embedder、contextual chunker / HyDE 拿到 llm。各策略元件之間**不互相 import**。`config.py` 的 `RAGConfig` 刻意只存字串不存物件，組裝責任全在 pipeline。

## 改 code 時必守的約定

- **新增一個策略** = 三步：(1) 在對應目錄加實作檔，類別上加 `@REGISTRY.register("名字")`；(2) 在該類別的 `__init__.py` 的 `.lazy(...)` 加一行；(3) 完成——UI 下拉選單自動出現。不要在 `__init__.py` 直接 `import` 實作模組（會破壞延遲載入）。

- **不要在頁面層 import 重函式庫**。`tests/test_lazy_imports.py` 守的就是這條：進「建立索引」頁時 torch/faiss/sentence_transformers 等不可出現在 `sys.modules`，只有建 pipeline 才付成本。`pipeline._make_reranker` / cross_encoder 的 import 都刻意延到使用時。

- **模型權重以名稱為 key 快取**（`@st.cache_resource`，見 `embedding/local.py` 的 `_load_model`、`rerank/cross_encoder.py`）。換配置但同模型名要重用同一個 model 物件，別重載 80MB~1GB 權重。`tests/test_model_caching.py` 守這條（用 `_FakeModel` 替換驗證只實例化一次）。

- **embed_query 與 embed_documents 分開**：BGE 等模型查詢端要加 prefix、文件端不加（見 `local.py` 的 `QUERY_PREFIX`）。新 embedder 沿用這個分工。

## 需要知道的限制

- **狀態全在記憶體**：`state.py` 的 `documents / pipelines / eval_results / chat_history` 存在 `st.session_state`，重開 Streamlit 就消失。持久化做法見 `vectorstore/chroma_store.py`、`qdrant_store.py` 內註解。
- **CPU/記憶體保護**：`torch_runtime.limit_torch_threads()` 在載 embedding/rerank 模型前限制 torch 執行緒（預設留 2 顆核給 server，避免 encode 吃滿 CPU 導致 websocket 斷線）；環境變數 `RAG_LAB_TORCH_THREADS` 可覆寫。`_STEmbedder` 的 `embed_documents` 用 `batch_size`（預設 16）分批編碼壓低記憶體峰值。
- **兩階段檢索**：開 reranker 時 retriever 撈 `final_k × 4` 候選（`build_index` 的 `fetch_k`），再由 cross-encoder 精排回 `final_k`。
- **`langchain-community` 鎖在 `>=0.3,<0.4`**：ragas 仍 import 已被 0.4 移除的 `chat_models.vertexai`。出現 `No module named 'langchain_community.chat_models.vertexai'` 就是版本跑掉了。
- **contextual chunking 每個 chunk 呼叫一次 LLM**，大文件建索引極慢；評估時間 ≈ 配置數 × 題數 ×（1 答 + 數次裁判），先用 3–5 題小規模試。
