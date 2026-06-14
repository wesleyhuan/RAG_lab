import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULTS = {
    "documents": [],      # list[Document]
    "pipelines": {},      # name -> RAGPipeline（已建好索引）
    "eval_results": {},   # name -> DataFrame（RAGAS 結果）
    "chat_history": {},   # name -> list[{"role","content"}]
}


def init_state():
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
