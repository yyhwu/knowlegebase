"""
检索质量评估 — MRR / NDCG@k / Precision@k
输入：评估数据集 + 检索结果 → 输出指标
"""
from __future__ import annotations

import math
from collections import defaultdict

from app.pipeline.retriever import search as rag_search


def evaluate_retrieval(
    eval_data: list[dict],
    top_k: int = 5,
) -> dict:
    """
    评估检索质量

    对每条 eval query:
      1. 执行检索，获取 top_k 结果
      2. 与 ground_truth_doc_ids 对比
      3. 计算 MRR / NDCG@k / Precision@k
    """
    mrr_sum = 0.0
    ndcg_sum = 0.0
    precision_sum = 0.0
    total = 0

    for item in eval_data:
        query = item["query"]
        gt_ids = set(item.get("ground_truth_doc_ids", []))
        relevance_scores = item.get("relevance_scores")  # optional per-doc scores
        if not gt_ids:
            continue

        # 执行检索
        docs = rag_search(query=query, top_k=top_k, expand_context=False)

        # ── MRR: 第一个相关文档的排名倒数 ──
        for rank, doc in enumerate(docs, 1):
            if doc["doc_id"] in gt_ids:
                mrr_sum += 1.0 / rank
                break

        # ── Precision@k ──
        hits = sum(1 for d in docs if d["doc_id"] in gt_ids)
        precision_sum += hits / top_k

        # ── NDCG@k ──
        ndcg = _compute_ndcg(docs, gt_ids, relevance_scores, top_k)
        ndcg_sum += ndcg

        total += 1

    if total == 0:
        return {"error": "没有有效的评估数据"}

    return {
        "total_queries": total,
        "top_k": top_k,
        "MRR": round(mrr_sum / total, 4),
        f"Precision@{top_k}": round(precision_sum / total, 4),
        f"NDCG@{top_k}": round(ndcg_sum / total, 4),
    }


def _compute_ndcg(
    docs: list[dict],
    gt_ids: set[str],
    relevance_scores: dict | None,
    k: int,
) -> float:
    """计算 NDCG@k"""
    # 实际 DCG
    dcg = 0.0
    for rank, doc in enumerate(docs[:k], 1):
        if doc["doc_id"] in gt_ids:
            rel = relevance_scores.get(doc["doc_id"], 1) if relevance_scores else 1
            dcg += rel / math.log2(rank + 1)

    # 理想 DCG（所有相关文档排在最前面）
    idcg = 0.0
    if relevance_scores:
        sorted_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    else:
        sorted_rels = [1] * min(len(gt_ids), k)
    for rank, rel in enumerate(sorted_rels, 1):
        idcg += rel / math.log2(rank + 1)

    return min(dcg / idcg, 1.0) if idcg > 0 else 0.0
