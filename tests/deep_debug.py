"""深度诊断：精确模拟 ingest.py 的 _extract_pdf_fallback 对 rag切分.pdf 的行为"""
import os, io, sys
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, r"c:\Users\Administrator\Desktop\knowleggeBase")

path = r"C:\Users\Administrator\Desktop\测试文件\rag切分.pdf"
with open(path, "rb") as f:
    content = f.read()

print(f"=== rag切分.pdf: {len(content)} bytes ===")

# 精确复制 _extract_pdf_fallback 的逻辑
from pypdf import PdfReader
reader = PdfReader(io.BytesIO(content))
all_text_pypdf = []
for page in reader.pages:
    t = (page.extract_text() or "").strip()
    if t:
        all_text_pypdf.append(t)
total_len_pypdf = sum(len(t) for t in all_text_pypdf)
print(f"[pypdf] pages_with_text={len(all_text_pypdf)}/{len(reader.pages)}, total_chars={total_len_pypdf}")

import pdfplumber
with pdfplumber.open(io.BytesIO(content)) as pdf:
    all_text_pl = []
    for page in pdf.pages:
        t = (page.extract_text() or "").strip()
        if t:
            all_text_pl.append(t)
        tables = page.extract_tables()
        for table in tables:
            if table:
                rows = [" | ".join(str(c or "") for c in row) for row in table]
                all_text_pl.append("[表格]\n" + "\n".join(rows))
total_len_pl = sum(len(t) for t in all_text_pl)
print(f"[pdfplumber] fragments={len(all_text_pl)}/{len(pdf.pages)}p, total_chars={total_len_pl}")

# 判断走哪条路径
if total_len_pypdf >= 100:
    print(f"\n>>> PATH: pypdf (returning early, {total_len_pypdf} chars)")
    text = "\n\n".join(all_text_pypdf)
elif total_len_pl >= 100:
    print(f"\n>>> PATH: pdfplumber (returning, {total_len_pl} chars)")
    text = "\n\n".join(all_text_pl)
else:
    print(f"\n>>> PATH: OCR needed (pypdf={total_len_pypdf}<100, pdfplumber={total_len_pl}<100)")
    print("(OCR skipped in this test for speed)")
    import sys; sys.exit(0)

print(f"\nText preview (first 300 chars):")
print(repr(text[:300]))
print(f"\nText preview (last 200 chars):")
print(repr(text[-200:]))

# 测试分片
from app.pipeline.splitter import split_text, _has_md_headers
print(f"\nhas_md_headers: {_has_md_headers(text)}")
chunks = split_text(text, "faq")
print(f"Chunks: {len(chunks)}")
for c in chunks[:5]:
    print(f"  [{c.chunk_idx}] {len(c.text)} chars")
