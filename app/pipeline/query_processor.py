"""
Query 预处理 — 轻量默认 + 按需 LLM 增强
策略：默认只做关键词提取，检测到触发条件才调用 LLM
"""
from __future__ import annotations

import re

from app.core.llm import get_llm
from app.core.embeddings import get_embedder


# ── 触发检测条件 ──────────────────────────

def _is_fuzzy(query: str) -> bool:
    """检测模糊查询：代词、笼统表述"""
    fuzzy_patterns = [
        r'那个|这个|它|他|她|那些|这些',
        r'怎么弄|怎么做|搞一下|帮我看看',
        r'有没有[^？?]{0,5}的',  # "有没有相关的"
    ]
    return any(re.search(p, query) for p in fuzzy_patterns)


def _is_complex(query: str) -> bool:
    """检测复杂查询：含多个问题/对比/条件"""
    indicators = [
        len(query) > 40,  # 长查询
        bool(re.search(r'[?？]', query)),  # 含问号
        bool(re.search(r'区别|对比|比较|还是|以及|另外|同时', query)),
        query.count('？') + query.count('?') >= 2,  # 多个问句
    ]
    return sum(indicators) >= 2


def _needs_context(history: list[dict] | None) -> bool:
    """检测是否需要上下文补全"""
    if not history:
        return False
    last_user = [m for m in history if m.get("role") == "user"]
    if not last_user:
        return False
    return len(last_user) >= 2  # 至少2轮用户输入


# ── LLM 增强方法 ──────────────────────────

def _llm_hyde(query: str) -> str:
    """
    HyDE: 让 LLM 生成一个假设答案，用答案做向量检索
    解决"查询词不在文档中但语义相关"的问题
    例: "怎么报销" → LLM 生成假设答案"出差后填报销单..."
        → 这个答案的 embedding 更接近文档里的"费用核销流程"
    """
    llm = get_llm()
    prompt = (
        f"请根据常识，用一段话回答以下问题。不要担心对错，只需要生成一个合理的假设答案：\n\n"
        f"问题：{query}\n\n假设答案："
    )
    try:
        return llm.invoke(prompt)[:300]
    except Exception:
        return query  # 降级：原样返回


def _llm_decompose(query: str) -> list[str]:
    """拆解复杂查询为多个子问题"""
    llm = get_llm()
    prompt = (
        f"把以下复杂问题拆解为最少的独立子问题（每行一个，只输出问题不要编号）：\n\n{query}"
    )
    try:
        result = llm.invoke(prompt)
        sub_queries = [q.strip() for q in result.split('\n') if q.strip() and len(q.strip()) > 2]
        return sub_queries if sub_queries else [query]
    except Exception:
        return [query]


def _llm_context_completion(query: str, history: list[dict]) -> str:
    """上下文补全：把指代词替换为具体实体"""
    llm = get_llm()
    history_text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
        for m in history[-6:]
    )
    prompt = (
        f"基于对话历史，把用户最新问题的指代词替换成具体内容。只输出改写后的问题，不要解释：\n\n"
        f"{history_text}\n用户: {query}\n改写后:"
    )
    try:
        result = llm.invoke(prompt)
        return result.strip()[:200] if result.strip() else query
    except Exception:
        return query


# ── 主入口 ────────────────────────────────

def process_query(
    query: str,
    history: list[dict] | None = None,
) -> list[str]:
    """
    Query 预处理主入口 — 轻量默认 + 按需增强

    返回: 检索用的 query 列表（单 query 或多子问题）

    策略:
      1. 永远做关键词提取（零 LLM 调用）
      2. 有 history → 上下文补全（+1 LLM）
      3. 模糊查询 → HyDE 生成假答案（+1 LLM）
      4. 复杂查询 → 拆解子问题（+1 LLM）
      最多额外 2 次 LLM 调用（上下文补全复用 HyDE 结果）
    """
    results: list[str] = []

    # 步骤 1：上下文补全（多轮对话）
    effective_query = query
    if _needs_context(history):
        effective_query = _llm_context_completion(query, history)

    # 步骤 2：意图判断
    is_fuzzy = _is_fuzzy(effective_query)
    is_complex = _is_complex(effective_query)

    if is_complex:
        # 复杂查询：拆解为多个子问题
        sub_queries = _llm_decompose(effective_query)
        results.extend(sub_queries)
    elif is_fuzzy:
        # 模糊查询：HyDE 生成假答案 + 原问题
        hyde_answer = _llm_hyde(effective_query)
        results.extend([effective_query, hyde_answer])
    else:
        # 简单查询：原样 + 用 embedding 检索
        results.append(effective_query)

    return results


def get_hybrid_queries(query: str, history: list[dict] | None = None) -> dict:
    """
    返回混合检索所需的全部查询配置
    {dense_queries: [...],  bm25_query: str,  original: str,  rewritten: str}

    bm25_query 用于 Milvus 稀疏向量检索（BM25 Function 自动分词 + 评分），
    由原始 query + query_expansion 同义词扩展拼接而成。
    """
    processed = process_query(query, history)

    # BM25 查询文本：原始 query + jieba 分词同义词扩展 → Milvus 内置分词器处理
    from app.pipeline.query_expansion import expand_keywords
    expanded_kws = expand_keywords(query)
    bm25_parts = [query]
    if expanded_kws:
        bm25_parts.extend(expanded_kws)
    bm25_query = " ".join(bm25_parts)

    return {
        "dense_queries": processed,
        "bm25_query": bm25_query,
        "original": query,
        "rewritten": processed[0] if processed else query,
    }
