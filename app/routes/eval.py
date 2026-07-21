"""
/api/eval — RAG 评估接口
"""
from fastapi import APIRouter

from app.models.schemas import ChatResponse
from app.pipeline.eval_dataset import generate_eval_queries, load_eval_dataset
from app.pipeline.eval_retrieval import evaluate_retrieval
from app.pipeline.eval_generation import evaluate_generation

router = APIRouter()


@router.post("/eval/generate-dataset")
async def api_generate_dataset(num_per_doc: int = 3):
    """自动生成评估数据集"""
    data = generate_eval_queries(num_per_doc=num_per_doc)
    return {"message": f"已生成 {len(data)} 条评估数据", "count": len(data)}


@router.get("/eval/retrieval")
async def api_eval_retrieval(top_k: int = 5):
    """评估检索质量（MRR / NDCG / Precision）"""
    eval_data = load_eval_dataset()
    if not eval_data:
        return {"error": "没有评估数据，请先调用 POST /api/eval/generate-dataset"}
    results = evaluate_retrieval(eval_data, top_k=top_k)
    return results


@router.get("/eval/generation")
async def api_eval_generation(top_k: int = 3):
    """评估生成质量（Faithfulness / Answer Relevance）"""
    eval_data = load_eval_dataset()
    if not eval_data:
        return {"error": "没有评估数据，请先调用 POST /api/eval/generate-dataset"}
    results = evaluate_generation(eval_data, top_k=top_k)
    return results


@router.post("/eval/full")
async def api_eval_full(top_k_for_retrieval: int = 5, top_k_for_gen: int = 3):
    """完整评估：检索 + 生成"""
    eval_data = load_eval_dataset()
    if not eval_data:
        return {"error": "没有评估数据，请先调用 POST /api/eval/generate-dataset"}

    retrieval = evaluate_retrieval(eval_data, top_k=top_k_for_retrieval)
    generation = evaluate_generation(eval_data, top_k=top_k_for_gen)

    return {
        "dataset_size": len(eval_data),
        "retrieval": retrieval,
        "generation": generation,
    }
