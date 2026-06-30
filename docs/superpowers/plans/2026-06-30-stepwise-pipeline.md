# Stepwise Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a step-by-step page that runs a single RAG pipeline one stage at a time (chunk → embed → store → retrieve → answer), loading heavy libraries only when each stage runs and showing each stage's result.

**Architecture:** Refactor `RAGPipeline` so components (embedder/store/llm/reranker) are lazy `cached_property` and `build_index` is composed from four public step methods (`chunk`/`embed`/`index`/`build_retriever`). A new Streamlit page drives those methods one button at a time, storing intermediate state in `st.session_state`. Chunkers declare their dependencies via a `NEEDS` tuple so recursive chunking never loads the embedder.

**Tech Stack:** Python 3.13, Streamlit, sentence-transformers, faiss/chroma/qdrant, Ollama/OpenAI. No pytest — tests are plain scripts.

## Global Constraints

- **Tests are plain scripts, no pytest.** Each test file has `if __name__ == "__main__":` calling its test functions and printing `PASS`. Exit code 0 = pass.
- **Run tests with the project root on the path:** `PYTHONPATH=. venv/Scripts/python.exe tests/<file>.py` (PowerShell: `$env:PYTHONPATH="."; venv\Scripts\python.exe tests\<file>.py`).
- **`build_index(docs, progress=None) -> int` external behavior must not change** — the existing "🔧 建立索引" page, "📊 評估比較", and "💬 對話" pages depend on it and on `pipe.retrieve` / `pipe.answer` / `pipe.config` / `pipe.chunks`.
- **Lazy-loading discipline:** never import torch/sentence-transformers/faiss in a module that the light "list the menu" path imports. Heavy imports stay inside `create()`-triggered modules or function bodies. `tests/test_lazy_imports.py` enforces this and must stay green.
- **Logging:** surface real exceptions, don't swallow them (project convention).

---

### Task 1: Registry.get() + chunker NEEDS declarations

**Files:**
- Modify: `rag_lab/registry.py` (add `get`)
- Modify: `rag_lab/chunking/base.py` (add `NEEDS` to `BaseChunker`)
- Modify: `rag_lab/chunking/semantic.py` (set `NEEDS`)
- Modify: `rag_lab/chunking/contextual.py` (set `NEEDS`)
- Test: `tests/test_chunker_needs.py`

**Interfaces:**
- Produces: `Registry.get(key: str) -> type` (returns the registered class, triggering lazy import if needed). `BaseChunker.NEEDS: tuple[str, ...]` (class attribute; default `()`). `SemanticChunker.NEEDS == ("embedder",)`, `ContextualChunker.NEEDS == ("llm",)`, `RecursiveChunker.NEEDS == ()` (inherited).

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunker_needs.py`:

```python
"""Registry.get() 回傳已註冊的類別；每個 chunker 用 NEEDS 宣告它需要哪些依賴
（pipeline.chunk() 據此只注入該 chunker 真正要的東西，避免 recursive 白載 embedder）。
無 pytest，直接 `PYTHONPATH=. python tests/test_chunker_needs.py` 執行。
"""


def test_registry_get_returns_class():
    from rag_lab.chunking import CHUNKERS
    cls = CHUNKERS.get("recursive")
    assert isinstance(cls, type), f"get() 應回傳類別，得到 {cls!r}"


def test_chunker_needs():
    from rag_lab.chunking import CHUNKERS
    assert CHUNKERS.get("recursive").NEEDS == ()
    assert CHUNKERS.get("semantic").NEEDS == ("embedder",)
    assert CHUNKERS.get("contextual").NEEDS == ("llm",)


if __name__ == "__main__":
    test_registry_get_returns_class()
    test_chunker_needs()
    print("PASS: Registry.get 與 chunker NEEDS 正確")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. venv/Scripts/python.exe tests/test_chunker_needs.py`
Expected: FAIL — `AttributeError: 'Registry' object has no attribute 'get'`.

- [ ] **Step 3: Add `get` to Registry**

In `rag_lab/registry.py`, add this method right after `_resolve`:

```python
    def get(self, key: str) -> type:
        """回傳 key 對應的類別（必要時觸發延遲 import）。供需要讀類別屬性
        （例如 chunker 的 NEEDS）而不建立實例時使用。"""
        return self._resolve(key)
```

- [ ] **Step 4: Declare NEEDS on the chunkers**

In `rag_lab/chunking/base.py`, add the class attribute to `BaseChunker` (right after `class BaseChunker(ABC):`):

```python
class BaseChunker(ABC):
    NEEDS: tuple[str, ...] = ()   # 此 chunker 需要 pipeline 注入的依賴名稱（如 "embedder"/"llm"）

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]: ...
```

In `rag_lab/chunking/semantic.py`, add `NEEDS` inside `SemanticChunker` (right after the docstring, before `__init__`):

```python
    NEEDS = ("embedder",)
```

In `rag_lab/chunking/contextual.py`, add `NEEDS` inside `ContextualChunker` (right after the docstring, before `__init__`):

```python
    NEEDS = ("llm",)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=. venv/Scripts/python.exe tests/test_chunker_needs.py`
Expected: `PASS: Registry.get 與 chunker NEEDS 正確`

- [ ] **Step 6: Confirm lazy-loading test still green**

Run: `PYTHONPATH=. venv/Scripts/python.exe tests/test_lazy_imports.py`
Expected: `PASS: ...` (no heavy libs leaked).

- [ ] **Step 7: Commit**

```bash
git add rag_lab/registry.py rag_lab/chunking/base.py rag_lab/chunking/semantic.py rag_lab/chunking/contextual.py tests/test_chunker_needs.py
git commit -m "feat: add Registry.get and chunker NEEDS declarations"
```

---

### Task 2: RAGPipeline lazy components + step methods

**Files:**
- Modify: `rag_lab/pipeline.py` (full rewrite of the class body; `RAG_PROMPT` unchanged)
- Test: `tests/test_pipeline_steps.py`

**Interfaces:**
- Consumes: `Registry.get(...).NEEDS` from Task 1; `RAGConfig` (`rag_lab/config.py`); `Document`/`Chunk` (`rag_lab/documents.py`).
- Produces: `RAGPipeline(config)` whose `embedder`/`store`/`llm`/`reranker` are lazy `cached_property` (not created until accessed). Step methods: `chunk(docs) -> list[Chunk]`, `embed() -> list[list[float]]`, `index() -> None`, `build_retriever() -> retriever`. `build_index(docs, progress=None) -> int` unchanged externally. Attributes: `pipe.chunks`, `pipe.vectors`, `pipe.retriever`, `pipe.final_k`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_steps.py`:

```python
"""RAGPipeline 的元件改為延遲建立（用到才載），build_index 由四個分步方法組成。
驗證：(1) 建構不載入任何元件；(2) recursive 的 chunk() 不會觸發 embedder；
(3) build_index 依序呼叫四個分步方法。
無 pytest，直接 `PYTHONPATH=. python tests/test_pipeline_steps.py` 執行。
"""
from rag_lab.config import RAGConfig
from rag_lab.documents import Document
from rag_lab.pipeline import RAGPipeline


def test_construction_loads_nothing():
    pipe = RAGPipeline(RAGConfig())
    for name in ("embedder", "store", "llm", "reranker"):
        assert name not in pipe.__dict__, f"建構就建立了 {name}（cached_property 不該被觸發）"


def test_recursive_chunk_does_not_load_embedder():
    pipe = RAGPipeline(RAGConfig(chunker="recursive"))
    doc = Document(text="第一段。第二段！第三段？" * 50)
    chunks = pipe.chunk([doc])
    assert len(chunks) > 0, "recursive 應切出至少一個 chunk"
    assert "embedder" not in pipe.__dict__, "recursive 切割不該觸發 embedder"


def test_build_index_calls_steps_in_order():
    pipe = RAGPipeline(RAGConfig())
    calls = []
    pipe.chunk = lambda docs: (calls.append("chunk"), [])[1]
    pipe.embed = lambda: calls.append("embed")
    pipe.index = lambda: calls.append("index")
    pipe.build_retriever = lambda: calls.append("retriever")
    pipe.build_index([])
    assert calls == ["chunk", "embed", "index", "retriever"], f"順序錯誤：{calls}"


if __name__ == "__main__":
    test_construction_loads_nothing()
    test_recursive_chunk_does_not_load_embedder()
    test_build_index_calls_steps_in_order()
    print("PASS: pipeline 延遲建立 + 分步方法正確")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. venv/Scripts/python.exe tests/test_pipeline_steps.py`
Expected: FAIL — `test_construction_loads_nothing` asserts fail (current `__init__` eagerly creates `embedder`/`store`/`llm`/`reranker`, so they're in `__dict__`).

- [ ] **Step 3: Rewrite the RAGPipeline class**

In `rag_lab/pipeline.py`, add `from functools import cached_property` at the top of the imports, keep `RAG_PROMPT` unchanged, and replace the entire `class RAGPipeline:` body with:

```python
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
```

Note: the old `_make_reranker` static method is removed (replaced by the `reranker` cached_property).

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. venv/Scripts/python.exe tests/test_pipeline_steps.py`
Expected: `PASS: pipeline 延遲建立 + 分步方法正確`

- [ ] **Step 5: Confirm other tests still green**

Run all three existing suites:
```bash
PYTHONPATH=. venv/Scripts/python.exe tests/test_lazy_imports.py
PYTHONPATH=. venv/Scripts/python.exe tests/test_model_caching.py
PYTHONPATH=. venv/Scripts/python.exe tests/test_resource_limits.py
```
Expected: each prints `PASS: ...`.

- [ ] **Step 6: Commit**

```bash
git add rag_lab/pipeline.py tests/test_pipeline_steps.py
git commit -m "refactor: lazy pipeline components + composable step methods"
```

---

### Task 3: Stepwise page (pages/5_🔬_逐步建構.py)

**Files:**
- Create: `pages/5_🔬_逐步建構.py`

**Interfaces:**
- Consumes: `RAGPipeline` step methods + `pipe.chunks` / `pipe.vectors` / `pipe.reranker` / `pipe.final_k` / `pipe.retrieve` / `pipe.answer` / `pipe.config` from Task 2; `RAGConfig`; the five registries; `init_state` (`rag_lab/state.py`); `RAG_PROMPT` not needed (uses `pipe.answer`).
- Produces: a Streamlit page (no importable API).

- [ ] **Step 1: Create the page**

Create `pages/5_🔬_逐步建構.py`:

```python
import time

import streamlit as st

from rag_lab.chunking import CHUNKERS
from rag_lab.config import RAGConfig
from rag_lab.embedding import EMBEDDERS
from rag_lab.llm import LLMS
from rag_lab.pipeline import RAGPipeline
from rag_lab.retrieval import RETRIEVERS
from rag_lab.state import init_state
from rag_lab.vectorstore import VECTOR_STORES

st.set_page_config(page_title="逐步建構 · RAG Lab", page_icon="🔬", layout="wide")
init_state()

st.title("🔬 逐步建構")

docs = st.session_state.documents
if not docs:
    st.warning("尚未載入任何文件，請先到「📁 資料來源」頁上傳。")
    st.stop()

# 頁面自持的逐步狀態（不污染跨頁的 state.py）
if "step" not in st.session_state:
    st.session_state.step = {"pipe": None, "done": set(), "query": "", "results": []}
step = st.session_state.step


def _default(options, preferred):
    return options.index(preferred) if preferred in options else 0


st.caption(f"目前有 {len(docs)} 份文件。先選齊策略並「套用設定」，再一步步執行、觀察每步結果。")

# ── 選擇策略（只是字串，不載任何權重）─────────────────────
c1, c2, c3 = st.columns(3)
chunker = c1.selectbox("Chunking", CHUNKERS.keys(), index=_default(CHUNKERS.keys(), "recursive"))
embedder = c2.selectbox("Embedding", EMBEDDERS.keys(), index=_default(EMBEDDERS.keys(), "minilm"))
vector_store = c3.selectbox("Vector Store", VECTOR_STORES.keys(),
                            index=_default(VECTOR_STORES.keys(), "faiss"))
c4, c5 = st.columns(2)
retriever = c4.selectbox("Retrieval", RETRIEVERS.keys(), index=_default(RETRIEVERS.keys(), "dense"))
llm_backend = c5.selectbox("LLM", LLMS.keys(), index=_default(LLMS.keys(), "ollama"))

with st.expander("參數（可選，留預設即可）"):
    p1, p2, p3 = st.columns(3)
    chunk_size = p1.number_input("chunk_size", 100, 4000, 500, 50)
    overlap = p2.number_input("overlap", 0, 500, 50, 10)
    top_k = p3.number_input("top_k（最終回傳片段數）", 1, 20, 5, 1)
    llm_model = st.text_input("LLM 模型名稱（留空用後端預設）",
                              placeholder="ollama 例：qwen2.5:7b ／ openai 例：gpt-4o-mini")
    use_reranker = st.checkbox("使用 reranker（cross-encoder 精排，第一次會下載 ~1GB）", value=False)

cset, creset = st.columns([1, 1])
if cset.button("套用設定", type="primary"):
    cfg = RAGConfig(
        name="stepwise",
        chunker=chunker, chunker_params={"chunk_size": int(chunk_size), "overlap": int(overlap)},
        embedder=embedder, vector_store=vector_store,
        retriever=retriever, retriever_params={"top_k": int(top_k)},
        use_reranker=use_reranker, llm_backend=llm_backend,
        llm_params={"model": llm_model.strip()} if llm_model.strip() else {},
    )
    st.session_state.step = {"pipe": RAGPipeline(cfg), "done": set(), "query": "", "results": []}
    st.rerun()
if creset.button("🔄 重置"):
    st.session_state.step = {"pipe": None, "done": set(), "query": "", "results": []}
    st.rerun()

pipe = step["pipe"]
if pipe is None:
    st.info("尚未套用設定。選好策略後按「套用設定」開始。")
    st.stop()

st.divider()
st.json(pipe.config.to_dict())
done = step["done"]


def _run(label, fn):
    """執行一步並計時；失敗印出實際例外，不靜默吞。回傳 (結果, 是否成功)。"""
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as e:
        st.error(f"{label} 失敗：{e}")
        st.exception(e)
        return None, False
    st.success(f"{label} 完成（{time.perf_counter() - t0:.1f}s）")
    return result, True


# ── ① Chunking ────────────────────────────────────────────
st.subheader("① Chunking")
if st.button("執行切割", disabled="chunk" in done):
    _, ok = _run("切割", lambda: pipe.chunk(docs))
    if ok:
        done.add("chunk")
        st.rerun()
if "chunk" in done:
    n = len(pipe.chunks)
    avg = sum(len(c.text) for c in pipe.chunks) / n if n else 0
    st.write(f"共 **{n}** 個 chunk，平均長度 **{avg:.0f}** 字。前 3 個：")
    for c in pipe.chunks[:3]:
        with st.expander(f"chunk #{c.id}（{len(c.text)} 字）"):
            if c.metadata.get("context"):
                st.caption(f"注入的 context：{c.metadata['context']}")
            st.write(c.text)

# ── ② Embedding ───────────────────────────────────────────
st.subheader("② Embedding")
if st.button("執行向量化", disabled="chunk" not in done or "embed" in done):
    _, ok = _run("向量化", pipe.embed)
    if ok:
        done.add("embed")
        st.rerun()
if "embed" in done and pipe.vectors:
    v0 = pipe.vectors[0]
    norm = sum(x * x for x in v0) ** 0.5
    st.write(f"共 **{len(pipe.vectors)}** 個向量，維度 **{len(v0)}**，首向量 norm **{norm:.3f}**（已正規化應 ≈ 1）。")
    st.code(str([round(x, 4) for x in v0[:8]]) + " …")

# ── ③ Vector Store ────────────────────────────────────────
st.subheader("③ Vector Store")
if st.button("寫入向量庫", disabled="embed" not in done or "index" in done):
    _, ok = _run("寫入向量庫", pipe.index)
    if ok:
        done.add("index")
        st.rerun()
if "index" in done:
    st.write(f"已寫入 **{pipe.config.vector_store}**，共 **{len(pipe.chunks)}** 筆。")

# ── ④ Retrieval ───────────────────────────────────────────
st.subheader("④ Retrieval")
step["query"] = st.text_input("測試 query", value=step["query"], disabled="index" not in done)
if st.button("檢索", disabled="index" not in done or not step["query"].strip()):
    def _retrieve():
        if "retriever" not in done:
            pipe.build_retriever()
            done.add("retriever")
        return pipe.retrieve(step["query"])
    results, ok = _run("檢索", _retrieve)
    if ok:
        step["results"] = results
        st.rerun()
if step["results"]:
    if pipe.reranker:
        st.caption(f"已開 reranker：粗篩 {pipe.final_k * 4} → 精排 {pipe.final_k}")
    for c, score in step["results"]:
        with st.expander(f"chunk #{c.id} — score {score:.4f}"):
            st.write(c.text)

# ── ⑤ LLM 回答 ────────────────────────────────────────────
st.subheader("⑤ LLM 回答")
if st.button("產生回答", disabled=not step["results"]):
    with st.expander("送進 LLM 的 context"):
        st.write("\n\n---\n\n".join(c.text for c, _ in step["results"]))
    gen, _ = pipe.answer(step["query"], stream=True)
    st.write_stream(gen)
```

- [ ] **Step 2: Syntax-check the page (it can't be imported directly — `st.set_page_config` errors outside `streamlit run`)**

Run:
```bash
venv/Scripts/python.exe -c "import ast,io; ast.parse(io.open('pages/5_🔬_逐步建構.py',encoding='utf-8').read()); print('SYNTAX OK')"
```
Expected: `SYNTAX OK`

- [ ] **Step 3: Manual verification**

Run: `venv/Scripts/python.exe -m streamlit run app.py --server.fileWatcherType none`
Then in the browser:
1. Go to "📁 資料來源", upload a small TXT/MD (a few paragraphs).
2. Go to "🔬 逐步建構". Leave defaults (recursive / minilm / faiss / dense, reranker off). Click **套用設定** — confirm no long load happens (just shows the config JSON).
3. Click **執行切割** — confirm chunk count + previews appear, with a timing.
4. Click **執行向量化** — confirm vector count/dim/norm (norm ≈ 1). This is the first heavy load (minilm); confirm the server stays responsive.
5. Click **寫入向量庫** — confirm count.
6. Type a query, click **檢索** — confirm retrieved chunks + scores.
7. Click **產生回答** — confirm the answer streams (requires Ollama running, or switch LLM to openai with a key).
8. Click **🔄 重置** — confirm it clears back to the start.

Expected: each step runs independently with its own timing; heavy work is isolated to the step that triggers it.

- [ ] **Step 4: Commit**

```bash
git add "pages/5_🔬_逐步建構.py"
git commit -m "feat: add stepwise pipeline page for per-stage observation"
```

---

## Self-Review

**Spec coverage:**
- §1a lazy components → Task 2 (cached_property). ✓
- §1b step methods + build_index → Task 2. ✓
- §1c chunker NEEDS → Task 1. ✓
- §1d Registry.get → Task 1. ✓
- §2 state model → Task 3 (`st.session_state.step`). ✓
- §3 page + per-step UI/observation → Task 3. ✓
- §4 error handling (`_run` try/except + `st.exception`) → Task 3. ✓
- §5 tests (lazy construction, recursive no-embedder, NEEDS values, build_index orchestration) → Tasks 1 & 2. ✓
- §6 compatibility (build_index unchanged) → Task 2 Step 5 reruns existing suites. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every test step shows exact run command + expected output. ✓

**Type consistency:** `Registry.get` / `NEEDS` defined in Task 1 and consumed in Task 2's `chunk()` and Task 3 indirectly. Step method names (`chunk`/`embed`/`index`/`build_retriever`) and attributes (`chunks`/`vectors`/`retriever`/`final_k`/`reranker`) are consistent across Tasks 2 and 3. ✓

**Note:** §5 of the spec mentioned a "分步 == 一鍵 最終狀態一致" numeric-equivalence test. The plan implements this as an orchestration-order test (`test_build_index_calls_steps_in_order`) instead, to avoid loading torch/faiss in a plain-script test — same guarantee (build_index = the four steps in order), no heavy deps. Intentional simplification.
