from .base import LLMS, BaseLLM  # noqa: F401

# 延遲註冊：openai 會 import openai SDK，到 create() 才載入
LLMS.lazy(
    ollama="rag_lab.llm.ollama_llm",
    openai="rag_lab.llm.openai_llm",
)
