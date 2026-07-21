"""
Prompt 拼装 — System Prompt + 检索上下文 + 对话历史 + 当前查询
企业 RAG 版：去掉风格参考层，改为展示文档标题路径
"""
from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"


def _load_prompt(filename: str, **kwargs) -> str:
    path = _PROMPT_DIR / filename
    if not path.exists():
        return ""
    template = path.read_text(encoding="utf-8")
    return template.format(**kwargs)


def build_rag_prompt(
    query: str,
    docs: list[dict],
    history: list[dict] | None = None,
) -> str:
    """
    拼装 RAG Prompt（企业版）

    分层结构:
      1. System Prompt（角色定位 + 引用要求）
      2. 对话历史（最近 6 轮）
      3. 检索上下文（每个 doc 带标题路径和相关度）
      4. 当前用户查询
    """
    # 1. System Prompt
    system = _load_prompt(
        "system.txt",
        role_name="企业知识库助手",
        role_description="基于企业知识库提供准确、专业的回答，必须引用来源。",
    )

    # 2. 对话历史
    history_text = ""
    if history:
        lines = []
        for msg in history[-6:]:
            role_label = "用户" if msg.get("role") == "user" else "助手"
            lines.append(f"{role_label}: {msg.get('content', '')}")
        if lines:
            history_text = "【对话历史】\n" + "\n".join(lines)

    # 3. 检索上下文
    context_parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        heading = f" > {doc.get('heading_title')}" if doc.get('heading_title') else ""
        chunk_info = f"(含 {doc.get('chunk_count', 1)} 个片段)" if doc.get('chunk_count', 1) > 1 else ""
        context_parts.append(
            f"### 参考 {i} [{doc.get('doc_type', '')}]{heading} (相关度: {doc.get('score', 0):.2f}) {chunk_info}\n"
            f"{doc.get('text', '')}"
        )

    context_text = (
        "【知识库参考】请基于以下内容回答。每条参考末尾标注了来源类型和相关度，"
        "请优先采用高相关度的内容。如果以下内容不足以回答问题，请明确告知。\n\n"
        + "\n\n".join(context_parts)
    )

    # 4. 当前查询
    query_text = f"【用户问题】\n{query}"

    parts = [system]
    if history_text:
        parts.append(history_text)
    parts.append(context_text)
    parts.append(query_text)

    return "\n\n".join(parts)
