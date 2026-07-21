"""
检索器 — 密集向量 + BM25 稀疏向量双路混合检索 + Query 预处理 + Reranker + 缓存
企业 RAG 版
"""
from __future__ import annotations

from app.config import settings
from app.core.embeddings import get_embedder
from app.core.milvus import get_milvus_collection
from app.pipeline.query_processor import get_hybrid_queries
from app.pipeline.reranker import rerank as rerank_docs
from app.pipeline.cache import (
    get_cached_search, set_cached_search,
    get_cached_embedding, set_cached_embedding,
)


OUTPUT_FIELDS = [
    "doc_id", "chunk_idx", "text", "doc_type",
    "parent_idx", "heading_title", "heading_level", "metadata",
]


def build_filter_expr(
    doc_types: list[str] | None = None,
    department: str | None = None,
) -> str | None:
    parts: list[str] = []
    if doc_types:
        quoted = ", ".join(f'"{t}"' for t in doc_types)
        parts.append(f'doc_type in [{quoted}]')
    if department:
        parts.append(f'metadata["department"] == "{department}"')
    return " and ".join(parts) if parts else None


def _dense_search(client, query_text: str, top_k: int, filter_expr: str | None) -> list[dict]:
    """密集向量检索（单 query）"""
    embedder = get_embedder()
    query_vec = embedder.embed_query(query_text)
    results = client.search(
        collection_name=settings.collection_name,
        data=[query_vec],
        limit=top_k,
        filter=filter_expr or "",
        output_fields=OUTPUT_FIELDS,
        anns_field="embedding",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )
    hits = results[0] if results else []
    return [
        {"doc_id": h["entity"]["doc_id"], "chunk_idx": h["entity"]["chunk_idx"],
         "text": h["entity"]["text"], "doc_type": h["entity"]["doc_type"],
         "parent_idx": h["entity"].get("parent_idx"), "heading_title": h["entity"].get("heading_title", ""),
         "heading_level": h["entity"].get("heading_level", 0),
         "metadata": h["entity"].get("metadata"), "score": h["distance"],
         "_source": "dense"}
        for h in hits
    ]


def _multi_dense_search(queries: list[str], top_k: int, filter_expr: str | None) -> list[dict]:
    """多 query 密集检索（支持 HyDE / 拆解），去重合并"""
    seen = set()
    all_results: list[dict] = []
    per_query = max(3, top_k // len(queries))

    for q in queries:
        results = _dense_search(client=None, query_text=q, top_k=per_query, filter_expr=filter_expr)
        # ... need client; refactor below
        break
    return all_results


def _bm25_search(client, query_text: str, top_k: int, filter_expr: str | None) -> list[dict]:
    """BM25 稀疏向量检索 — 利用 Milvus Function 自动分词 + BM25 评分"""
    if not query_text or not query_text.strip():
        return []

    try:
        results = client.search(
            collection_name=settings.collection_name,
            data=[query_text],
            limit=top_k,
            filter=filter_expr or "",
            output_fields=OUTPUT_FIELDS,
            anns_field="sparse_vector",
            search_params={"metric_type": "IP"},
        )
        hits = results[0] if results else []
        return [
            {"doc_id": h["entity"]["doc_id"], "chunk_idx": h["entity"]["chunk_idx"],
             "text": h["entity"]["text"], "doc_type": h["entity"]["doc_type"],
             "parent_idx": h["entity"].get("parent_idx"), "heading_title": h["entity"].get("heading_title", ""),
             "heading_level": h["entity"].get("heading_level", 0),
             "metadata": h["entity"].get("metadata"), "score": h["distance"],
             "_source": "bm25"}
            for h in hits
        ]
    except Exception:
        return []


def _rrf_fusion(dense_results: list[dict], bm25_results: list[dict], k: int = 60, top_k: int = 10) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) 融合两路结果"""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, doc in enumerate(dense_results):
        key = f"{doc['doc_id']}_{doc['chunk_idx']}"
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        docs[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = f"{doc['doc_id']}_{doc['chunk_idx']}"
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        docs[key] = doc

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [dict(docs[k], score=scores[k]) for k in sorted_keys]


def search(
    query: str,
    top_k: int = 10,
    doc_types: list[str] | None = None,
    department: str | None = None,
    history: list[dict] | None = None,
    expand_context: bool = True,
) -> list[dict]:
    """
    混合检索 — 支持 Query 预处理（HyDE/拆解/上下文补全）

    流程:
      1. Query 预处理 → 密集 queries + BM25 查询文本
      2. 多 query 密集检索 (含 HyDE 假答案)
      3. BM25 稀疏向量检索 (Milvus Function 自动分词 + 评分)
      4. RRF 融合排序
      5. 可选：父文档上下文扩展
    """
    embedder = get_embedder()
    client = get_milvus_collection()
    filter_expr = build_filter_expr(doc_types, department)

    # ── Query 预处理 ──
    qc = get_hybrid_queries(query, history)
    dense_queries = qc["dense_queries"]  # 可能含原问题 + HyDE 答案 + 子问题
    bm25_query = qc["bm25_query"]        # 扩展后的 BM25 查询文本

    # ── 密集检索（多 query，去重合并）──
    dense_results: list[dict] = []
    seen_dense = set()
    per_q = max(5, (top_k * 2) // max(len(dense_queries), 1))
    for q in dense_queries:
        vec = embedder.embed_query(q)
        results = client.search(
            collection_name=settings.collection_name,
            data=[vec],
            limit=per_q,
            filter=filter_expr or "",
            output_fields=OUTPUT_FIELDS,
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        for hit in (results[0] if results else []):
            key = hit["entity"]["doc_id"] + "_" + str(hit["entity"]["chunk_idx"])
            if key not in seen_dense:
                seen_dense.add(key)
                dense_results.append({
                    "doc_id": hit["entity"]["doc_id"],
                    "chunk_idx": hit["entity"]["chunk_idx"],
                    "text": hit["entity"]["text"],
                    "doc_type": hit["entity"]["doc_type"],
                    "parent_idx": hit["entity"].get("parent_idx"),
                    "heading_title": hit["entity"].get("heading_title", ""),
                    "heading_level": hit["entity"].get("heading_level", 0),
                    "metadata": hit["entity"].get("metadata"),
                    "score": hit["distance"],
                    "_source": "dense",
                })

    # ── BM25 稀疏向量检索 ──
    bm25_results = _bm25_search(client, bm25_query, top_k * 2, filter_expr)

    # ── RRF 融合 ──
    chunks = _rrf_fusion(dense_results, bm25_results, top_k=top_k)

    # ── Reranker 精排（DashScope gte-rerank）──
    if settings.reranker_provider and len(chunks) > 1:
        try:
            chunks = rerank_docs(query, chunks, top_n=settings.reranker_top_n)
        except Exception:
            pass  # 降级：reranker 失败时保留 RRF 排序

    if not expand_context:
        return chunks
    return _expand_to_parent_context(chunks)


def _expand_to_parent_context(chunks: list[dict]) -> list[dict]:
    """将同 doc 同 parent 的相邻小 chunk 合并为上下文片段"""
    groups: dict[tuple, list[dict]] = {}
    for c in chunks:
        key = (c["doc_id"], c.get("parent_idx") or -1)
        groups.setdefault(key, []).append(c)

    merged: list[dict] = []
    for key, group in groups.items():
        seen = set()
        unique = []
        for c in sorted(group, key=lambda x: x["chunk_idx"]):
            if c["text"] not in seen:
                seen.add(c["text"])
                unique.append(c)

        if len(unique) == 1:
            merged.append(_fmt(unique[0]))
        else:
            combined_text = "\n\n".join(c["text"] for c in unique)
            best = max(unique, key=lambda x: x["score"])
            merged.append({
                "doc_id": best["doc_id"], "text": combined_text, "score": best["score"],
                "doc_type": best["doc_type"], "heading_title": best["heading_title"],
                "heading_level": best["heading_level"], "chunk_count": len(unique),
                "metadata": best["metadata"],
            })

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


def _fmt(c: dict) -> dict:
    return {
        "doc_id": c["doc_id"], "text": c["text"], "score": c["score"],
        "doc_type": c["doc_type"], "heading_title": c.get("heading_title", ""),
        "heading_level": c.get("heading_level", 0), "chunk_count": 1,
        "metadata": c.get("metadata"),
    }
