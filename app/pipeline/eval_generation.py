"""
生成质量评估 — Faithfulness / Answer Relevance
使用 LLM-as-Judge 自动打分
"""
from __future__ import annotations

from app.core.llm import get_llm


def evaluate_faithfulness(answer: str, contexts: list[str]) -> dict:
    """
    忠实度评估：答案中的每个陈述是否有检索内容作为依据

    返回: {score: 0.0~1.0, reason: "..."}
    """
    llm = get_llm()
    ctx_text = "\n---\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts[:5]))

    prompt = f"""你是一个 RAG 评估专家。请判断以下"答案"中的每个陈述是否都能在"参考上下文"中找到依据。

参考上下文:
{ctx_text}

答案:
{answer}

请用以下 JSON 格式回答（只输出 JSON）：
{{
  "score": 0.0到1.0之间的分数（1.0=完全忠实，0.0=完全编造）,
  "reason": "简要说明扣分原因（如果满分则说'全部有依据'）"
}}"""

    try:
        raw = llm.invoke(prompt)
        import json
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"score": 0.5, "reason": "评估失败"}


def evaluate_answer_relevance(query: str, answer: str) -> dict:
    """
    答案相关性：生成的答案是否真正回答了用户的问题

    返回: {score: 0.0~1.0, reason: "..."}
    """
    llm = get_llm()

    prompt = f"""你是一个 RAG 评估专家。请判断以下"答案"是否直接、完整地回答了"用户问题"。

用户问题: {query}

答案: {answer}

请用 JSON 格式回答（只输出 JSON）：
{{
  "score": 0.0到1.0之间的分数（1.0=完美回答，0.0=完全无关）,
  "reason": "简要说明"
}}"""

    try:
        raw = llm.invoke(prompt)
        import json
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"score": 0.5, "reason": "评估失败"}


def evaluate_generation(eval_data: list[dict], top_k: int = 3) -> dict:
    """
    批量评估生成质量

    对每条 eval query:
      1. 先检索获取 context
      2. 让 LLM 生成答案
      3. 评估 Faithfulness + Answer Relevance
    """
    from app.pipeline.retriever import search as rag_search
    from app.core.llm import get_llm as _get_llm
    from app.pipeline.prompt import build_rag_prompt

    llm = _get_llm()
    faith_scores = []
    relevance_scores = []

    for item in eval_data:
        query = item["query"]
        ref_answer = item.get("reference_answer", "")

        # 检索
        docs = rag_search(query=query, top_k=top_k, expand_context=True)
        contexts = [d["text"] for d in docs]

        # 生成
        prompt = build_rag_prompt(query=query, docs=docs)
        answer = llm.invoke(prompt)

        # 评估
        f = evaluate_faithfulness(answer, contexts)
        r = evaluate_answer_relevance(query, answer)

        faith_scores.append(f.get("score", 0))
        relevance_scores.append(r.get("score", 0))

    if not faith_scores:
        return {"error": "无评估数据"}

    return {
        "total": len(faith_scores),
        "avg_faithfulness": round(sum(faith_scores) / len(faith_scores), 4),
        "avg_answer_relevance": round(sum(relevance_scores) / len(relevance_scores), 4),
    }
