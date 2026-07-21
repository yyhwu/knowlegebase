"""
Reranker 精排 — 对混合检索 top-k 结果用 cross-encoder 二次排序
支持两种模式: DashScope API (默认) / 本地模型 (预留)
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from app.config import settings


def _dashscope_rerank(query: str, documents: list[str], top_n: int) -> list[dict]:
    """
    使用 DashScope Reranker API 对文档重排序
    https://help.aliyun.com/document_detail/2787022.html
    """
    import dashscope
    from dashscope import TextReRank

    dashscope.api_key = settings.dashscope_api_key

    resp = TextReRank.call(
        model=settings.reranker_model_name,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
        return_documents=True,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Reranker API error: {resp.message}")

    results = []
    for item in resp.output.results:
        idx = item.get("index", 0)
        results.append({
            "index": idx,
            "text": documents[idx] if idx < len(documents) else "",
            "score": item.get("relevance_score", 0),
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]


def _local_rerank(query: str, documents: list[str], top_n: int) -> list[dict]:
    """本地 Reranker 桩（待集成 bge-reranker-large）"""
    # TODO: from FlagEmbedding import FlagReranker
    return [{"index": i, "text": d, "score": 1.0 - 0.01 * i} for i, d in enumerate(documents)][:top_n]


def rerank(
    query: str,
    documents: list[dict],  # 含 text 字段的检索结果
    top_n: int | None = None,
) -> list[dict]:
    """
    对检索结果重排序

    参数:
        query: 用户原始查询
        documents: [{"text": "...", ...}, ...] 格式的检索结果
        top_n: 最终返回数量，默认等于输入数量

    返回: 重排序后的 documents（score 更新为 rerank 分数）
    """
    if not documents:
        return []

    if top_n is None:
        top_n = len(documents)

    texts = [doc["text"] for doc in documents]

    try:
        if settings.reranker_provider == "dashscope":
            ranked = _dashscope_rerank(query, texts, top_n)
        elif settings.reranker_provider == "local":
            ranked = _local_rerank(query, texts, top_n)
        else:
            return documents[:top_n]
    except Exception:
        # 降级：返回原始排序
        return documents[:top_n]

    # 保持原始 metadata，更新 score
    result = []
    for item in ranked:
        idx = item["index"]
        if idx < len(documents):
            doc = dict(documents[idx])
            doc["score"] = item["score"]
            doc["_reranked"] = True
            result.append(doc)
    return result
