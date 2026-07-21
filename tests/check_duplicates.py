"""检查 Milvus 数据库状态 — 去重分析"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sys
sys.path.insert(0, r'c:\Users\Administrator\Desktop\knowleggeBase')

from app.core.milvus import get_milvus_collection
from app.config import settings
from collections import Counter

client = get_milvus_collection()

# 获取所有 chunk
r = client.query(
    collection_name=settings.collection_name,
    filter="",  # 不过滤
    output_fields=["doc_id", "chunk_idx", "doc_type", "text"],
    limit=9999,
)
print(f"总 chunk 数: {len(r)}")

# 按 doc_id 分组
doc_groups = {}
for item in r:
    did = item["doc_id"]
    doc_groups.setdefault(did, []).append(item)

print(f"独立文档数 (unique doc_id): {len(doc_groups)}")

# 分析每个文档的 chunk 数
chunk_counts = [len(v) for k, v in doc_groups.items()]
print(f"chunks/doc: min={min(chunk_counts)}, max={max(chunk_counts)}, avg={sum(chunk_counts)/len(chunk_counts):.1f}")

# 检查文本重复（前80字）
text_prefixes = []
for item in r:
    prefix = item["text"][:80].strip()
    text_prefixes.append(prefix)

prefix_count = Counter(text_prefixes)
duplicates = {k: v for k, v in prefix_count.items() if v > 1}
if duplicates:
    print(f"\n⚠️ 重复文本前缀: {len(duplicates)} 组")
    for prefix, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
        print(f"  [{count}x] {prefix[:60]}...")
else:
    print("\n✅ 无重复文本前缀")

# 显示各文档信息
print(f"\n{'='*60}")
print(f"文档列表:")
print(f"{'='*60}")
for did, chunks in sorted(doc_groups.items(), key=lambda x: -len(x[1])):
    doc_type = chunks[0]["doc_type"]
    first_text = chunks[0]["text"][:60].replace('\n', ' ')
    print(f"  {did[:8]}... type={doc_type}, chunks={len(chunks)}, text={first_text}...")
