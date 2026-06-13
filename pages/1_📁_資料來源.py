import streamlit as st

from rag_lab.ingestion import load_pdf, load_text, load_url
from rag_lab.state import init_state

st.set_page_config(page_title="資料來源 · RAG Lab", page_icon="📁", layout="wide")
init_state()

st.title("📁 資料來源")
st.caption("上傳 PDF / TXT / MD，或輸入網址抓取網頁。文件會存進記憶體，供「建立索引」頁使用。")


def _sources() -> set[str]:
    """目前已載入文件的 source 集合，用來判斷重複。"""
    return {d.metadata.get("source", "") for d in st.session_state.documents}


def _add(doc) -> bool:
    """加入一份文件，若 source 重複則跳過。回傳是否真的加入。"""
    if doc.metadata.get("source", "") in _sources():
        return False
    st.session_state.documents.append(doc)
    return True


# ── 1. 上傳檔案 ──────────────────────────────────────────────
st.subheader("1. 上傳檔案")
uploaded = st.file_uploader(
    "選擇檔案（可多選）",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("加入文件", type="primary", disabled=not uploaded):
    added, skipped, failed = 0, 0, []
    for f in uploaded:
        try:
            if f.name.lower().endswith(".pdf"):
                doc = load_pdf(f, name=f.name)
            else:
                raw = f.read()
                try:
                    content = raw.decode("utf-8")
                except UnicodeDecodeError:
                    content = raw.decode("utf-8", errors="replace")
                doc = load_text(content, name=f.name)
        except Exception as e:  # 解析失敗不影響其他檔案
            failed.append(f"{f.name}（{e}）")
            continue
        if not doc.text.strip():   # 例：掃描影像 PDF 擷取不到文字
            failed.append(f"{f.name}（擷取不到文字，可能是掃描影像 PDF）")
            continue
        if _add(doc):
            added += 1
        else:
            skipped += 1

    if added:
        st.success(f"已加入 {added} 份文件。")
    if skipped:
        st.warning(f"略過 {skipped} 份（source 名稱重複）。")
    if failed:
        st.error("以下檔案解析失敗：\n\n- " + "\n- ".join(failed))

# ── 2. 抓取網頁 ──────────────────────────────────────────────
st.subheader("2. 抓取網頁")
url = st.text_input("網址", placeholder="https://example.com/article")

if st.button("抓取網頁", disabled=not url.strip()):
    target = url.strip()
    if target in _sources():
        st.warning("這個網址已經載入過了。")
    else:
        try:
            doc = load_url(target)
        except Exception as e:
            st.error(f"抓取失敗：{e}")
        else:
            if doc.text:
                _add(doc)
                st.success(f"已加入網頁：{target}")
            else:
                st.warning("抓到的網頁沒有可用的正文內容。")

# ── 3. 已載入文件 ────────────────────────────────────────────
st.divider()
docs = st.session_state.documents
st.subheader(f"已載入文件（{len(docs)}）")

if not docs:
    st.info("尚未載入任何文件。")
else:
    for i, d in enumerate(docs):
        col_info, col_del = st.columns([0.9, 0.1])
        source = d.metadata.get("source", "（未命名）")
        col_info.markdown(f"**{source}** — {len(d.text):,} 字")
        if col_del.button("🗑️", key=f"del_{i}", help="刪除這份文件"):
            st.session_state.documents.pop(i)
            st.rerun()

    if st.button("全部清除"):
        st.session_state.documents = []
        st.rerun()
