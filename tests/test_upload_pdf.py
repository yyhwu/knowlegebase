import httpx, time, json, sys

pdf_path = "C:/Users/Administrator/Desktop/测试文件/rag优化.pdf"
print(f"上传: {pdf_path}")
start = time.time()

with open(pdf_path, "rb") as f:
    r = httpx.post(
        "http://localhost:8000/api/ingest",
        files={"file": ("rag优化.pdf", f, "application/pdf")},
        data={"doc_type": "sop", "department": "测试"},
        timeout=300,
    )

elapsed = time.time() - start
print(f"耗时: {elapsed:.1f}s, 状态: {r.status_code}")

if r.status_code == 200:
    d = json.loads(r.text)
    print(f"✅ 入库成功: {d.get('chunks', '?')} 个片段")
    print(f"   doc_id: {d.get('doc_id', '?')}")
else:
    print(f"❌ 失败: {r.text[:500]}")
