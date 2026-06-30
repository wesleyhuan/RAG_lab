"""限制 PyTorch 在 CPU 上開的執行緒數。

embedding / rerank 用 sentence-transformers 在 CPU 上跑時，預設會把所有核心
吃到 100%，讓 Streamlit server 線程搶不到 CPU、瀏覽器 websocket 斷線——
表面上像「server 停了」，其實多半還在算。這裡留幾顆核給 server 用。

預設留 2 顆核；可用環境變數 RAG_LAB_TORCH_THREADS 覆寫（例如機器很弱設成 1）。
"""
import logging
import os

logger = logging.getLogger(__name__)

_applied = False   # 只套用一次（set_num_threads 全域生效，重複呼叫無意義）


def _default_threads() -> int:
    cpu = os.cpu_count() or 4
    return max(1, cpu - 2)          # 留 2 顆核給 Streamlit server / 瀏覽器


def limit_torch_threads() -> None:
    global _applied
    if _applied:
        return

    import torch   # 延遲 import：別拖累「只列選單」的輕量路徑（見 test_lazy_imports）

    env = os.getenv("RAG_LAB_TORCH_THREADS")
    if env:
        try:
            n = max(1, int(env))
        except ValueError:
            logger.warning("RAG_LAB_TORCH_THREADS=%r 不是整數，改用預設值", env)
            n = _default_threads()
    else:
        n = _default_threads()

    torch.set_num_threads(n)
    _applied = True
    logger.info("torch CPU 執行緒限制為 %d（總核心數 %s）", n, os.cpu_count())
