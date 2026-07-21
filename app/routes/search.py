"""
/api/search — 密集 + BM25 双路混合检索
"""
from fastapi import APIRouter

from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.pipeline.retriever import search

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def api_search(req: SearchRequest):
    """混合检索：返回匹配的文档列表（密集向量 + BM25 关键词）"""
    results = search(
        query=req.query,
        top_k=req.top_k,
        doc_types=req.doc_types,
        department=req.department,
        expand_context=False,  # 纯检索不做上下文扩展
    )

    return SearchResponse(
        results=[
            SearchResult(
                doc_id=r["doc_id"],
                chunk_idx=r.get("chunk_idx", 0),
                text=r["text"],
                score=r["score"],
                doc_type=r["doc_type"],
                metadata=r.get("metadata"),
            )
            for r in results
        ],
        total=len(results),
    )
