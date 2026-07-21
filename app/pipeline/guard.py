"""
后置校验 Guard — 企业 RAG 版：长度截断 + 空回复兜底 + 引用提醒
（去掉动漫对话的「禁止词」检测）
"""
from __future__ import annotations

from app.config import settings


def apply_guard(answer: str) -> tuple[str, list[str]]:
    """
    对 LLM 输出做后置校验。

    返回: (校验后的 answer, 触发的告警列表)
    """
    warnings: list[str] = []

    # 1. 长度硬截断
    max_len = settings.guard_max_answer_length
    if len(answer) > max_len:
        warnings.append(f"回答过长 (>{max_len})，已截断")
        answer = answer[:max_len] + "…"

    # 2. 空回复兜底
    if not answer or not answer.strip():
        answer = "抱歉，根据现有知识库无法回答该问题，请尝试换个问法。"
        warnings.append("空回复，使用兜底回复")

    # 3. 幻觉风险标记：如果回答明显在编造但无引用依据
    hallucination_hints = [
        ("据我所知", "「据我所知」可能是编造，建议核实"),
        ("一般来说", "「一般来说」缺乏引用依据"),
        ("通常情况下", "「通常情况下」缺乏引用依据"),
    ]
    for hint, warn in hallucination_hints:
        if hint in answer:
            warnings.append(warn)

    return answer, warnings
