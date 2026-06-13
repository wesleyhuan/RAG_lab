import streamlit as st

from rag_lab.state import init_state

st.set_page_config(page_title="對話 · RAG Lab", page_icon="💬", layout="wide")
init_state()

st.title("💬 對話")

pipelines = st.session_state.pipelines
if not pipelines:
    st.warning("尚未建立任何配置，請先到「🔧 建立索引」頁建立至少一組。")
    st.stop()

# ── 選擇配置 ─────────────────────────────────────────────────
top = st.columns([0.7, 0.3])
name = top[0].selectbox("選擇配置", list(pipelines.keys()))
pipe = pipelines[name]

# 若這組配置已評估過，附上平均分數當參考
eval_df = st.session_state.eval_results.get(name)
if eval_df is not None:
    scores = [f"{m}={eval_df[m].mean():.3f}"
              for m in ["faithfulness", "answer_relevancy",
                        "context_precision", "context_recall"]
              if m in eval_df.columns]
    if scores:
        st.caption("此配置評估平均：" + " · ".join(scores))

history = st.session_state.chat_history.setdefault(name, [])
if top[1].button("清除這組對話", use_container_width=True):
    st.session_state.chat_history[name] = []
    st.rerun()

# ── 歷史訊息 ─────────────────────────────────────────────────
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 輸入與回答 ───────────────────────────────────────────────
if prompt := st.chat_input("輸入問題…"):
    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results = []
        collected: list[str] = []
        try:
            stream, results = pipe.answer(prompt, stream=True)

            def _tee(gen):
                for token in gen:
                    collected.append(token)
                    yield token

            answer = st.write_stream(_tee(stream))
        except Exception as e:
            # 串流中途斷線時保留已產生的部分文字，不要整段丟失
            partial = "".join(collected)
            answer = f"{partial}\n\n_（回答中斷：{e}）_" if partial else f"回答失敗：{e}"
            st.error(f"回答中斷：{e}")

        if results:
            with st.expander(f"檢索到的片段（{len(results)}）"):
                for i, (chunk, score) in enumerate(results, 1):
                    src = chunk.metadata.get("source", "（未命名）")
                    st.markdown(f"**{i}. {src}** — score={score:.4f}")
                    st.write(chunk.text)

    history.append({"role": "assistant", "content": answer})
