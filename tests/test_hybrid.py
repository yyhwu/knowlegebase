"""测试混合检索：密集 + BM25"""
from app.pipeline.splitter import split_structured
from app.core.embeddings import get_embedder
from app.core.milvus import get_milvus_collection, release_collection
from app.config import settings

# 测试文档（含精确关键词）
text = """# API 接口文档

## 3.1 用户认证
接口 POST /api/v2/auth/login，请求体包含 username 和 password。

## 3.2 数据查询
接口 GET /api/v2/data/query?endpoint=ERP-MOD-7，返回 JSON。

## 3.3 错误码
ERR_TIMEOUT: 请求超时，ERR_AUTH_FAILED: 认证失败。

## 常见问题
如何调用 ERP-MOD-7 接口？请参考 3.2 数据查询。"""

# 1. 分块
chunks, parents = split_structured(text, doc_type="spec")
print(f"[分块] {len(chunks)} 子chunk, {len(parents)} 父chunk")

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
    "sparse_vector": {},
} for c, emb in zip(chunks, embs)]
client.insert(collection_name=settings.collection_name, data=rows)
print(f"[入库] {len(rows)} 条\n")

# 3. 混合检索测试
import httpx

def ask(q):
    r = httpx.post("http://localhost:8000/api/chat", json={"query": q}, timeout=30)
    d = r.json()
    print(f"Q: {q}")
    print(f"A: {d['answer'][:150]}")
    print(f"来源: {d['sources']}\n")

# 语义检索："怎么登录"应该匹配"用户认证"
ask("怎么登录系统？")

# 精确关键词：ERP-MOD-7 是专有名词，BM25 应精准命中
ask("ERP-MOD-7 怎么调用？")

# 组合：错误码查询
ask("ERR_TIMEOUT 是什么意思？")

release_collection()
print("✅ 混合检索测试完成")
