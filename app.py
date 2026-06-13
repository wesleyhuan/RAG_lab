import streamlit as st

from rag_lab.state import init_state

st.set_page_config(page_title="RAG Lab", page_icon="🧪", layout="wide")
init_state()

st.title("🧪 RAG Lab — 可組合的 RAG 實驗平台")
st.markdown("""
依照左側頁面順序操作：

| 步驟 | 頁面 | 做什麼 |
|---|---|---|
| 1 | **📁 資料來源** | 上傳 PDF / TXT / MD，或輸入網址抓取網頁 |
| 2 | **🔧 建立索引** | 組合 chunking × embedding × vector store × retrieval × rerank，可建立多組配置 |
| 3 | **📊 評估比較** | 輸入測試問題，用 RAGAS 對所有配置跑分、並排比較 |
| 4 | **💬 對話** | 選一個（評估後勝出的）配置開始問答，可展開看檢索到的片段 |

**目前狀態**
""")

c1, c2, c3 = st.columns(3)
c1.metric("文件數", len(st.session_state.documents))
c2.metric("已建立配置", len(st.session_state.pipelines))
c3.metric("已評估配置", len(st.session_state.eval_results))
