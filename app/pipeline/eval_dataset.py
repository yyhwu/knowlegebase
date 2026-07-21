"""
RAG 评估数据集 — 从已入库文档中自动生成评估数据
使用 LLM 生成 (query, 相关文档, 参考答案) 三元组
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.llm import get_llm
from app.core.logging import get_logger
from app.core.milvus import get_milvus_collection
from app.config import settings

logger = get_logger(__name__)

_EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "eval_data"


def generate_eval_queries(num_per_doc: int = 3, output_file: str = "eval_dataset.json") -> list[dict]:
    """
    对知识库中的文档自动生成评估 query

    返回: [
      {
        "query": "用户问题",
        "ground_truth_doc_ids": ["doc_id_1", ...],
        "reference_answer": "参考答案",
        "relevance": 2  # 强相关
      }, ...
    ]
    """
    client = get_milvus_collection()
    llm = get_llm()

    # 1. 读取所有入库文档（按 doc_id 分组采样）
    results = client.query(
        collection_name=settings.collection_name,
        filter="chunk_idx == 0",  # 每个文档取第一条
        output_fields=["doc_id", "text", "doc_type"],
        limit=50,
    )

    eval_data: list[dict] = []
    for doc in results:
        doc_id = doc["doc_id"]
        doc_text = doc.get("text", "")[:2000]  # 截取前 2000 字
        doc_type = doc.get("doc_type", "")

        # 2. LLM 生成基于该文档的问题 + 答案
        prompt = f"""你是一个 RAG 评估数据生成器。请根据以下文档内容，生成 {num_per_doc} 个用户可能会问的问题。
对每个问题，用 JSON 数组格式输出，只输出 JSON，不要其他文字：
[
  {{"question": "问题1", "answer": "基于文档的参考答案1"}},
  {{"question": "问题2", "answer": "基于文档的参考答案2"}}
]

文档类型: {doc_type}
文档内容:
{doc_text}
"""
        try:
            raw = llm.invoke(prompt)
            # 提取 JSON 部分
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                qa_pairs = json.loads(raw[start:end])
                for qa in qa_pairs:
                    eval_data.append({
                        "query": qa["question"],
                        "ground_truth_doc_ids": [doc_id],
                        "reference_answer": qa["answer"],
                        "relevance": 2,  # 强相关
                        "doc_type": doc_type,
                    })
        except Exception:
            continue

    # 3. 保存
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = _EVAL_DIR / output_file
    path.write_text(json.dumps(eval_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("生成 %d 条评估数据 → %s", len(eval_data), path)
    return eval_data


def load_eval_dataset(filename: str = "eval_dataset.json") -> list[dict]:
    """加载已有的评估数据集"""
    path = _EVAL_DIR / filename
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
