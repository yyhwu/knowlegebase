"""端到端测试：导入 Markdown → 检索 → 对话"""
from app.pipeline.splitter import split_structured
from app.core.embeddings import get_embedder
from app.core.milvus import get_milvus_collection, release_collection
from app.config import settings

md_text = """# 员工手册

## 第一章 考勤制度
员工应按时上下班。每日工作时间：9:00-18:00，午休 12:00-13:00。
请假需提前一天提交申请，经直属领导批准。

## 第二章 报销流程
差旅报销需在出差结束后 3 个工作日内提交。
单次报销金额超过 5000 元须部门总监审批。

### 2.1 交通报销
高铁二等座及以下实报实销，飞机经济舱需提前申请。

### 2.2 住宿报销
一线城市住宿标准 500 元/晚，其他城市 350 元/晚。"""

# 1. 分块
chunks, parents = split_structured(md_text, doc_type="sop")
print(f"[分块] {len(chunks)} 个子chunk, {len(parents)} 个父chunk")

# 2. 向量化 + 入库
embedder = get_embedder()
client = get_milvus_collection()
texts = [c.text for c in chunks]
embs = embedder.embed_documents(texts)
rows = [{
    "doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "text": c.text,
    "embedding": emb, "doc_type": c.doc_type, "parent_idx": c.parent_idx or -1,
    "heading_title": c.heading_title or "", "heading_level": c.heading_level,
    "metadata": c.metadata,
} for c, emb in zip(chunks, embs)]
client.insert(collection_name=settings.collection_name, data=rows)
print(f"[入库] {len(rows)} 条")

# 3. 检索 + 对话
import httpx

resp = httpx.post("http://localhost:8000/api/chat", json={
    "query": "报销需要哪些单据？",
})
data = resp.json()
print(f"\n[问题] 报销需要哪些单据？")
print(f"[回答] {data['answer'][:200]}")
print(f"[来源] {data['sources']}")

resp2 = httpx.post("http://localhost:8000/api/chat", json={
    "query": "住宿报销标准是什么？",
})
data2 = resp2.json()
print(f"\n[问题] 住宿报销标准是什么？")
print(f"[回答] {data2['answer'][:200]}")
print(f"[来源] {data2['sources']}")

release_collection()
print("\n✅ 端到端测试完成")
