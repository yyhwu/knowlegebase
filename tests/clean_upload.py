"""上传干净文档到知识库"""
import os, sys
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import requests

BASE = r"C:\Users\Administrator\Desktop\测试文件"
API = "http://localhost:8000/api/ingest"

files_to_upload = [
    ("万字长文，细节满满~RAG的优化.txt", "faq"),
    ("大模型面试题剖析：RAG中的文本分割策略.txt", "faq"),
    ("大模型面试题剖析：微调与 RAG 技术的选用逻辑.txt", "faq"),
]

for fname, dtype in files_to_upload:
    path = os.path.join(BASE, fname)
    if not os.path.exists(path):
        print(f"NOT FOUND: {fname}")
        continue
    with open(path, "rb") as f:
        r = requests.post(API, files={"file": (fname, f, "text/plain")},
                         data={"doc_type": dtype, "source": fname})
    if r.status_code == 200:
        data = r.json()
        print(f"OK: {fname[:30]}... -> {data['chunks']} chunks")
    else:
        print(f"ERR [{r.status_code}]: {fname[:30]}... -> {r.text[:80]}")
