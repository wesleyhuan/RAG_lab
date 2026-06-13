import pandas as pd
import streamlit as st

from rag_lab.state import init_state

# 注意：evaluate_pipeline 會 import ragas（連帶一長串 langchain 相依），
# 在最上層 import 會讓整頁在還沒按下評估前就崩潰。故延後到實際要評估時才載入。

st.set_page_config(page_title="評估比較 · RAG Lab", page_icon="📊", layout="wide")
init_state()

st.title("📊 評估比較")

pipelines = st.session_state.pipelines
if not pipelines:
    st.warning("尚未建立任何配置，請先到「🔧 建立索引」頁建立至少一組。")
    st.stop()

st.caption(
    "輸入測試問題，用 RAGAS 對選定的配置跑分並排比較。"
    "提供標準答案才會計算 context_precision / context_recall。"
)

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ── 輸入 ─────────────────────────────────────────────────────
q_raw = st.text_area("測試問題（一行一題）", height=140,
                     placeholder="這份文件在講什麼？\n作者的主要結論是什麼？")
gt_raw = st.text_area("標準答案（選填，一行一題，需與問題數一致）", height=140)
configs = st.multiselect("要評估的配置", list(pipelines.keys()),
                         default=list(pipelines.keys()))

st.info("評估時間 ≈ 配置數 × 題數 ×（1 次回答 + 數次裁判呼叫）。先用 3–5 題小規模試。")

if st.button("開始評估", type="primary"):
    questions = [q.strip() for q in q_raw.splitlines() if q.strip()]
    ground_truths = [g.strip() for g in gt_raw.splitlines() if g.strip()] or None

    if not questions:
        st.error("請至少輸入一題。")
    elif ground_truths and len(ground_truths) != len(questions):
        st.error(f"標準答案數（{len(ground_truths)}）與問題數（{len(questions)}）不一致。")
    elif not configs:
        st.error("請至少選一個配置。")
    else:
        try:
            from rag_lab.evaluation.ragas_eval import evaluate_pipeline
        except Exception as e:
            st.error(
                "無法載入 RAGAS 評估模組，請檢查相依套件版本：\n\n"
                f"`{type(e).__name__}: {e}`"
            )
            st.stop()

        for nm in configs:
            try:
                with st.status(f"評估「{nm}」…", expanded=True) as status:
                    df = evaluate_pipeline(pipelines[nm], questions,
                                           ground_truths, progress=status.write)
                    status.update(label=f"「{nm}」完成", state="complete")
                st.session_state.eval_results[nm] = df
            except Exception as e:
                st.error(f"「{nm}」評估失敗：{e}")

# ── 結果 ─────────────────────────────────────────────────────
results = st.session_state.eval_results
if results:
    st.divider()
    st.subheader("並排比較（各指標平均，越高越好）")

    summary = {}
    for nm, df in results.items():
        summary[nm] = {m: df[m].mean() for m in METRIC_COLS if m in df.columns}
    summary_df = pd.DataFrame(summary).T
    # 某配置在沒有標準答案的情況下評估時，context_* 欄會是 NaN（代表「未量測」），
    # 用 — 顯示以免被誤讀成 0 分。
    st.dataframe(summary_df.style.format("{:.3f}", na_rep="—"),
                 use_container_width=True)

    for nm, df in results.items():
        with st.expander(f"{nm} — 逐題明細"):
            st.dataframe(df, use_container_width=True)

    if st.button("清除評估結果"):
        st.session_state.eval_results = {}
        st.rerun()
