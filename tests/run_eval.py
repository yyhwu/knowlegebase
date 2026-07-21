"""一键 RAG 评估脚本：生成数据集 → 检索评估 → 生成评估"""
import os, sys, json
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import requests

BASE_URL = "http://localhost:8000"

# ── Step 1: 上传必须的文档（如果还没上传）──
print("=" * 60)
print("STEP 0: Check knowledge base")
print("=" * 60)

r = requests.post(f"{BASE_URL}/api/search", json={"query": "RAG 文本分割", "top_k": 2})
if r.status_code == 200:
    data = r.json()
    print(f"  Search test: {data.get('total', 0)} hits")
else:
    print(f"  Search failed: {r.status_code}")

# ── Step 2: 生成评估数据集 ──
print("\n" + "=" * 60)
print("STEP 1: Generate eval dataset (LLM will create questions)")
print("=" * 60)

r = requests.post(f"{BASE_URL}/api/eval/generate-dataset?num_per_doc=2")
if r.status_code == 200:
    data = r.json()
    print(f"  Generated: {data.get('count', 0)} eval items")
else:
    print(f"  Failed: {r.status_code} {r.text[:200]}")

# ── Step 3: 检索评估 ──
print("\n" + "=" * 60)
print("STEP 2: Retrieval evaluation (MRR, NDCG, Precision)")
print("=" * 60)

r = requests.get(f"{BASE_URL}/api/eval/retrieval?top_k=5")
if r.status_code == 200:
    data = r.json()
    if "error" in data:
        print(f"  Error: {data['error']}")
    else:
        print(f"  Total queries: {data.get('total_queries', 0)}")
        print(f"  MRR:           {data.get('MRR', 'N/A')}")
        print(f"  Precision@5:   {data.get('Precision@5', 'N/A')}")
        print(f"  NDCG@5:        {data.get('NDCG@5', 'N/A')}")
else:
    print(f"  Failed: {r.status_code} {r.text[:200]}")

# ── Step 4: 生成评估 ──
print("\n" + "=" * 60)
print("STEP 3: Generation evaluation (Faithfulness, Relevance)")
print("=" * 60)

r = requests.get(f"{BASE_URL}/api/eval/generation?top_k=3")
if r.status_code == 200:
    data = r.json()
    if "error" in data:
        print(f"  Error: {data['error']}")
    else:
        print(f"  Total:              {data.get('total', 0)}")
        print(f"  avg_faithfulness:   {data.get('avg_faithfulness', 'N/A')}")
        print(f"  avg_answer_relevance: {data.get('avg_answer_relevance', 'N/A')}")
else:
    print(f"  Failed: {r.status_code} {r.text[:200]}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
