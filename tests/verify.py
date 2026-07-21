"""全流程验证脚本"""
from app.config import settings
from app.core.milvus import get_milvus_collection, release_collection
from app.core.embeddings import get_embedder

print("=" * 50)
print("RAG 全流程测试")
print("=" * 50)

# 1. Milvus 连接 + 自动创建 Collection
print("\n[1/4] 连接 Milvus...")
client = get_milvus_collection()
print(f"  Collection: {settings.collection_name}")
print(f"  exists: {client.has_collection(settings.collection_name)}")
print(f"  indexes: {client.list_indexes(settings.collection_name)}")

# 2. 测试 Embedding
print("\n[2/4] 测试 Embedding...")
embedder = get_embedder()
test_vec = embedder.embed_query("测试数据")
print(f"  向量维度: {len(test_vec)}")

# 3. 测试插入
print("\n[3/4] 测试数据插入...")
client.insert(
    collection_name=settings.collection_name,
    data=[{
        "doc_id": "test-001",
        "chunk_idx": 0,
        "text": "这是一条测试数据，用于验证 RAG 系统。",
        "embedding": test_vec,
        "doc_type": "sop",
        "metadata": {"source": "test", "priority": "low"},
    }]
)
print("  插入成功")

# 4. 测试检索
print("\n[4/4] 测试向量检索...")
results = client.search(
    collection_name=settings.collection_name,
    data=[test_vec],
    limit=2,
    output_fields=["doc_id", "text", "doc_type"],
    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
)
hits = results[0]
print(f"  检索到 {len(hits)} 条结果")
for hit in hits:
    entity = hit.get("entity", {})
    text_preview = entity.get("text", "")[:50]
    print(f"  - [{entity.get('doc_type')}] {text_preview}... score={hit.get('distance', 0):.4f}")

# 5. 启动 FastAPI
print("\n" + "=" * 50)
print("全流程测试通过! 请运行:")
print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
print("=" * 50)

release_collection()
