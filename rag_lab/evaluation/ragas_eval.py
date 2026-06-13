"""RAGAS 評估。四個指標的意義：
- faithfulness        ：答案有多「忠於」檢索到的內容（抓幻覺）
- answer_relevancy    ：答案有多切題
- context_precision   ：檢索到的內容裡，有用的排得多前面（需要標準答案）
- context_recall      ：標準答案需要的資訊，檢索內容涵蓋了多少（需要標準答案）

前兩個沒有標準答案也能算；後兩個要提供 ground truth。
RAGAS 本身要用一個 LLM 當「裁判」——建議用比受測 LLM 更強的模型當裁判。
"""
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (answer_relevancy, context_precision,
                           context_recall, faithfulness)


def _judge_backends(llm_backend: str, llm_params: dict):
    """建立 RAGAS 用的裁判 LLM + embeddings（跟 pipeline 用同一個後端）"""
    if llm_backend == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        return (LangchainLLMWrapper(ChatOpenAI(
                    model=llm_params.get("model", "gpt-4o-mini"))),
                LangchainEmbeddingsWrapper(OpenAIEmbeddings(
                    model="text-embedding-3-small")))
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    return (LangchainLLMWrapper(ChatOllama(
                model=llm_params.get("model", "qwen2.5:7b"), num_ctx=8192)),
            LangchainEmbeddingsWrapper(OllamaEmbeddings(
                model="nomic-embed-text")))   # 需要先 ollama pull nomic-embed-text


def evaluate_pipeline(pipeline, questions: list[str],
                      ground_truths: list[str] | None = None, progress=None):
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for i, q in enumerate(questions):
        if progress:
            progress(f"回答第 {i + 1}/{len(questions)} 題…")
        answer, results = pipeline.answer(q)
        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append([c.text for c, _ in results])
        rows["ground_truth"].append(ground_truths[i] if ground_truths else "")

    metrics = [faithfulness, answer_relevancy]
    if ground_truths:
        metrics += [context_precision, context_recall]
    else:
        rows.pop("ground_truth")

    llm, emb = _judge_backends(pipeline.config.llm_backend,
                               pipeline.config.llm_params)
    if progress:
        progress("RAGAS 指標計算中（會多次呼叫裁判 LLM，請耐心等）…")
    result = evaluate(Dataset.from_dict(rows), metrics=metrics,
                      llm=llm, embeddings=emb)
    return result.to_pandas()
