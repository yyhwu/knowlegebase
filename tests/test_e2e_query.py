"""端到端：Query 预处理全场景测试"""
from app.pipeline.splitter import split_structured
from app.core.embeddings import get_embedder
from app.core.milvus import get_milvus_collection, release_collection
from app.config import settings
import httpx, json

# 准备测试文档
text = """# 企业管理制度

## 考勤制度
员工每日工作时间9:00-18:00，午休12:00-13:00。迟到30分钟内扣50元。

## 报销流程
差旅报销3个工作日内提交，需交通票据和住宿发票，超5000元须总监审批。
一线城市住宿标准500元/晚，其他350元/晚。

## 技术规范
接口ERR_TIMEOUT表示请求超时，ERR_AUTH_FAILED表示认证失败，ERR_RATE_LIMIT表示频率超限。"""

chunks, parents = split_structured(text, doc_type="sop")
e = get_embedder()
c = get_milvus_collection()
embs = e.embed_documents([ch.text for ch in chunks])
c.insert(collection_name=settings.collection_name, data=[{
    "doc_id": ch.doc_id, "chunk_idx": ch.chunk_idx, "text": ch.text,
    "embedding": emb, "doc_type": ch.doc_type,
    "parent_idx": ch.parent_idx or -1, "heading_title": ch.heading_title or "",
    "heading_level": ch.heading_level, "metadata": ch.metadata, "sparse_vector": {},
} for ch, emb in zip(chunks, embs)])
print(f"入库 {len(chunks)} 条\n")

# 测试场景
def test_scenario(name, query, history=None):
    print(f"=== {name} ===")
    print(f"Query: {query}")
    body = {"query": query}
    if history:
        body["history"] = history
    r = httpx.post("http://localhost:8000/api/chat", json=body, timeout=60)
    if r.status_code == 200:
        d = r.json()
        print(f"Answer: {d['answer'][:200]}")
        print(f"Sources: {len(d['sources'])}")
    else:
        print(f"ERROR: {r.status_code} {r.text[:200]}")
    print()

# 场景1: 简单查询（0次额外LLM）
test_scenario("简单", "报销标准是多少")

# 场景2: 模糊查询（触发 HyDE，+1 LLM）
test_scenario("模糊", "那个东西怎么搞的")

# 场景3: 多轮上下文补全（触发 context_completion，+1 LLM）
test_scenario("多轮上下文", "它的错误码是什么意思", [
    {"role": "user", "content": "ERR_TIMEOUT 出现了怎么办？"},
    {"role": "assistant", "content": "ERR_TIMEOUT 表示请求超时，请检查网络连接。"},
])

# 场景4: 复杂查询（触发分解，+1 LLM）
test_scenario("复杂", "考勤迟到的处罚是什么，报销超过5000元需要什么审批")

release_collection()
print("DONE")
