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
    t0 = time.perf_counter()
    try:
        gen, _ = pipe.answer(step["query"], stream=True)
        st.write_stream(gen)
    except Exception as e:
        st.error(f"產生回答失敗：{e}")
        st.exception(e)
    else:
        st.caption(f"完成（{time.perf_counter() - t0:.1f}s）")
