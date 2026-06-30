import streamlit as st

from rag_lab.chunking import CHUNKERS
from rag_lab.config import RAGConfig
from rag_lab.embedding import EMBEDDERS
from rag_lab.llm import LLMS
from rag_lab.pipeline import RAGPipeline
from rag_lab.retrieval import RETRIEVERS
from rag_lab.state import init_state
from rag_lab.vectorstore import VECTOR_STORES

st.set_page_config(page_title="建立索引 · RAG Lab", page_icon="🔧", layout="wide")
init_state()

st.title("🔧 建立索引")

docs = st.session_state.documents
if not docs:
    st.warning("尚未載入任何文件，請先到「📁 資料來源」頁上傳。")
    st.stop()

st.caption(f"目前有 {len(docs)} 份文件可用。組合各種策略建立一組（或多組）配置，稍後在「評估比較」頁跑分。")

# ── 選擇策略 ─────────────────────────────────────────────────
pipelines = st.session_state.pipelines
name = st.text_input("配置名稱", value=f"config-{len(pipelines) + 1}")

def _default(options, preferred):
    """預設選輕量、零下載的策略；該選項不存在時退回第一個。
    （registry.keys() 是字母排序，預設值會是 bge / chroma / contextual 這種重的組合，
     第一次建索引就要下載 ~1GB 並對每個 chunk 呼叫 LLM，故在此明確指定輕量預設。）"""
    return options.index(preferred) if preferred in options else 0


chunker_keys = CHUNKERS.keys()
embedder_keys = EMBEDDERS.keys()
store_keys = VECTOR_STORES.keys()
retriever_keys = RETRIEVERS.keys()
llm_keys = LLMS.keys()

c1, c2, c3 = st.columns(3)
chunker = c1.selectbox("Chunking", chunker_keys,
                       index=_default(chunker_keys, "recursive"))
embedder = c2.selectbox("Embedding", embedder_keys,
                        index=_default(embedder_keys, "minilm"))
vector_store = c3.selectbox("Vector Store", store_keys,
                            index=_default(store_keys, "faiss"))

c4, c5 = st.columns(2)
retriever = c4.selectbox("Retrieval", retriever_keys,
                         index=_default(retriever_keys, "dense"))
llm_backend = c5.selectbox("LLM", llm_keys,
                           index=_default(llm_keys, "ollama"))

with st.expander("參數（可選，留預設即可）"):
    p1, p2, p3 = st.columns(3)
    chunk_size = p1.number_input("chunk_size", 100, 4000, 500, 50)
    overlap = p2.number_input("overlap", 0, 500, 50, 10)
    top_k = p3.number_input("top_k（最終回傳片段數）", 1, 20, 5, 1)
    llm_model = st.text_input(
        "LLM 模型名稱（留空用後端預設）",
        placeholder="ollama 例：qwen2.5:7b ／ openai 例：gpt-4o-mini",
    )
    use_reranker = st.checkbox(
        "使用 reranker（cross-encoder 精排，第一次會下載 ~1GB 模型）", value=False
    )

# ── 建立 ─────────────────────────────────────────────────────
if st.button("建立索引", type="primary"):
    if not name.strip():
        st.error("請輸入配置名稱。")
    elif name in pipelines:
        st.error(f"配置名稱「{name}」已存在，請換一個。")
    else:
        cfg = RAGConfig(
            name=name,
            chunker=chunker,
            chunker_params={"chunk_size": int(chunk_size), "overlap": int(overlap)},
            embedder=embedder,
            vector_store=vector_store,
            retriever=retriever,
            retriever_params={"top_k": int(top_k)},
            use_reranker=use_reranker,
            llm_backend=llm_backend,
            llm_params={"model": llm_model.strip()} if llm_model.strip() else {},
        )
        try:
            with st.status("建立索引中…", expanded=True) as status:
                status.write("初始化元件（embedder / vector store / llm / reranker）…")
                pipe = RAGPipeline(cfg)
                n_chunks = pipe.build_index(docs, progress=status.write)
                status.update(label=f"完成：{n_chunks} 個 chunks", state="complete")
        except Exception as e:
            st.error(f"建立失敗：{e}")
        else:
            st.session_state.pipelines[name] = pipe
            st.success(f"配置「{name}」已建立（{n_chunks} 個 chunks）。")

# ── 已建立的配置 ─────────────────────────────────────────────
st.divider()
st.subheader(f"已建立的配置（{len(pipelines)}）")

if not pipelines:
    st.info("尚未建立任何配置。")
else:
    for nm, pipe in list(pipelines.items()):
        with st.expander(f"{nm} — {len(pipe.chunks)} chunks"):
            st.json(pipe.config.to_dict())
            if st.button("刪除這組配置", key=f"del_pipe_{nm}"):
                del st.session_state.pipelines[nm]
                st.session_state.eval_results.pop(nm, None)
                st.session_state.chat_history.pop(nm, None)
                st.rerun()
