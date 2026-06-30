# 逐步建構 Pipeline — 設計文件

日期：2026-06-30

## 背景與目標

「🔧 建立索引」頁用一個按鈕一次跑完 chunk → embed → store → retriever，在資源吃緊的機器上會出現 CPU/記憶體被吃滿、Streamlit server websocket 斷線（看起來像「server 停了」）。已先做的緩解：限制 torch 執行緒（`torch_runtime`）、embedding 分批（`batch_size`）。

本功能再加一個**逐步觀察單一 pipeline** 的頁面：把流程拆成可單獨觸發的步驟，每步只載入該步需要的重函式庫、執行後顯示結果，讓使用者：

1. **隔離資源**——重函式庫只在對應步驟按下去時才載入，不再一次全載。
2. **觀察每步**——看 chunk 數/預覽、向量維度、檢索片段+分數、LLM 回答，哪步是瓶頸一目了然。
3. **失敗不重來**——某步爆掉，修正後單獨重試，不必重跑前面。

## 範圍

- **In scope**：單一 pipeline 的逐步執行與觀察（含一次性問答）。
- **Out of scope**：多組配置比較、RAGAS 評估整合、多輪對話（沿用既有「📊 評估比較」「💬 對話」頁，本功能不動它們）。

## 設計

### 1. `RAGPipeline` 重構（`rag_lab/pipeline.py`）

#### 1a. 元件改延遲建立

目前 `__init__` 一建立就 `EMBEDDERS.create(...)`（minilm 會立刻載入 ~90MB 權重）。改成 `functools.cached_property`，用到才載、載一次：

```python
from functools import cached_property

def __init__(self, config: RAGConfig):
    self.config = config
    self.chunks: list[Chunk] = []
    self.vectors: list[list[float]] | None = None
    self.retriever = None
    self.final_k = config.retriever_params.get("top_k", 5)

@cached_property
def embedder(self): return EMBEDDERS.create(self.config.embedder, **self.config.embedder_params)

@cached_property
def store(self): return VECTOR_STORES.create(self.config.vector_store)

@cached_property
def llm(self): return LLMS.create(self.config.llm_backend, **self.config.llm_params)

@cached_property
def reranker(self):
    if not self.config.use_reranker:
        return None
    from rag_lab.rerank import CrossEncoderReranker   # 延遲 import，重
    return CrossEncoderReranker(self.config.reranker_model)
```

`_make_reranker` 靜態方法移除（由 `reranker` cached_property 取代）。

#### 1b. `build_index` 拆成四個分步方法

```python
def chunk(self, docs: list[Document]) -> list[Chunk]:
    needs = CHUNKERS.get(self.config.chunker).NEEDS        # 只注入該 chunker 真正要的依賴
    deps = {name: getattr(self, name) for name in needs}
    chunker = CHUNKERS.create(self.config.chunker, **deps, **self.config.chunker_params)
    self.chunks = chunker.chunk_all(docs)
    return self.chunks

def embed(self) -> list[list[float]]:
    self.vectors = self.embedder.embed_documents([c.text for c in self.chunks])
    return self.vectors

def index(self) -> None:
    self.store.add(self.chunks, self.vectors)

def build_retriever(self):
    fetch_k = self.final_k * 4 if self.reranker else self.final_k
    params = {**self.config.retriever_params, "top_k": fetch_k}
    self.retriever = RETRIEVERS.create(
        self.config.retriever, embedder=self.embedder, store=self.store,
        chunks=self.chunks, llm=self.llm, **params)
    return self.retriever

def build_index(self, docs, progress=None) -> int:   # 一鍵流程：行為與現在一致
    def log(msg):
        if progress: progress(msg)
    log(f"切割文件中（{self.config.chunker}）…");      self.chunk(docs)
    log(f"共 {len(self.chunks)} 個 chunks，向量化中（{self.config.embedder}）…"); self.embed()
    self.index()
    log(f"建立 retriever（{self.config.retriever}）…"); self.build_retriever()
    return len(self.chunks)
```

`retrieve()` / `answer()` 不變。

#### 1c. Chunker 宣告依賴（`NEEDS`）

避免 recursive 的切割步驟白白載入 embedder：

- `BaseChunker.NEEDS: tuple[str, ...] = ()`
- `SemanticChunker.NEEDS = ("embedder",)`
- `ContextualChunker.NEEDS = ("llm",)`

#### 1d. Registry 新增 public `get`

`Registry.get(key) -> type`：等同既有的私有 `_resolve`，供 `chunk()` 讀 `NEEDS`。`_resolve` 改成呼叫 `get`（或直接公開）。

### 2. 狀態模型

新頁自持狀態，不動跨頁的 `state.py`：

```python
if "step" not in st.session_state:
    st.session_state.step = {"pipe": None, "done": set(),
                             "query": "", "results": [], "answer": ""}
```

- `pipe`：`RAGPipeline`，本身持有 `chunks / vectors / retriever`。
- `done`：已完成步驟集合 `{"chunk","embed","index","retriever"}`。
- 重函式庫只在 `if st.button(...)` 區塊內執行；其餘 rerun 僅顯示已存結果。
- 「🔄 重置」清空 `step` 重來。

### 3. 頁面 `pages/5_🔬_逐步建構.py`

- 無文件時 `st.warning(...)` + `st.stop()`（同第 2 頁）。
- **頂部**：沿用第 2 頁的策略選單（chunker / embedder / store / retriever / llm + 參數 + `use_reranker`）。「套用設定」→ 建 `RAGConfig` + `RAGPipeline(cfg)`（不載入任何權重），重置 `done`。
- **五段**，每段按鈕僅在前一步完成後可按，**每步顯示耗時**（`time.perf_counter`）：

| 步 | 動作 | 觀察 |
|---|---|---|
| ① Chunking | `pipe.chunk(docs)` | chunk 數、平均長度、前 3 個預覽（contextual 顯示注入的 context） |
| ② Embedding | `pipe.embed()` | 向量數、維度、首向量前幾維 + norm |
| ③ Vector Store | `pipe.index()` | store 類型、已索引數 |
| ④ Retrieval | query 輸入 → `pipe.build_retriever()`（首次）→ `pipe.retrieve(query)` | 片段 + 分數；開 reranker 標示「粗篩 4×k → 精排 k」 |
| ⑤ LLM 回答 | `pipe.answer(query, stream=True)` | 串流回答 + 可展開看送入的 context |

頁碼用 5（接在「💬 對話」後）；不重排既有頁面。

### 4. 錯誤處理

- 每步包 `try/except`，`st.error` 印出實際例外訊息（不靜默吞，符合專案 logging 偏好）。失敗時不加入 `done`，可單獨重試。

### 5. 測試（`tests/test_pipeline_steps.py`，純腳本，`PYTHONPATH=.`）

用輕量假元件（避免載 torch/faiss），驗證：

1. **延遲載入**：`RAGPipeline(cfg)` 建構後 `"embedder" not in pipe.__dict__`（cached_property 未觸發）。
2. **recursive 不觸發 embedder**：`pipe.chunk(docs)`（recursive）後 `"embedder" not in pipe.__dict__`，且 `pipe.chunks` 非空。
3. **NEEDS 正確**：`CHUNKERS.get("semantic").NEEDS == ("embedder",)`、`("contextual").NEEDS == ("llm",)`、`("recursive").NEEDS == ()`。
4. **分步 == 一鍵**：對同一份 docs，依序 `chunk/embed/index/build_retriever` 後的 `chunks`、retriever 行為與 `build_index` 一致。

### 6. 相容性

- `build_index` 對外行為不變；「🔧 建立索引」頁、「📊 評估比較」「💬 對話」頁照舊。
- 已掃描全專案：外部僅第 2 頁讀 `pipe.chunks`（仍為一般屬性）；其餘 `.embedder/.store/.llm` 都在 retriever/chunker/store 自身物件上，與 pipeline cached_property 無關。皆為讀取，無 setter 衝突。
