"""检查当前数据库状态"""
import os, sys
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, r'c:\Users\Administrator\Desktop\knowleggeBase')

from app.core.milvus import get_milvus_collection
from app.config import settings

client = get_milvus_collection()

# 检查 collection 是否存在
stats = client.get_collection_stats(settings.collection_name)
print(f"Collection: {settings.collection_name}")
print(f"Row count: {stats.get('row_count', 0)}")

# 查看文档列表
if stats.get('row_count', 0) > 0:
    r = client.query(
        collection_name=settings.collection_name,
        filter="chunk_idx == 0",
        output_fields=["doc_id", "doc_type", "text", "metadata"],
        limit=50,
    )
    print(f"\nUnique docs (chunk_idx=0): {len(r)}")
    for d in r:
        source = d.get("metadata", {}).get("source", "?")
        text_preview = d["text"][:60].replace('\n', ' ')
        print(f"  {d['doc_id'][:8]}... type={d['doc_type']} source={source[:40]}")
        print(f"    {text_preview}...")
