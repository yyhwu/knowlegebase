"""快速诊断 rag切分.pdf 问题"""
import os, sys, io
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, r'c:\Users\Administrator\Desktop\knowleggeBase')

path = r"C:\Users\Administrator\Desktop\测试文件\rag切分.pdf"

# 模拟 ingest 流程
with open(path, 'rb') as f:
    content = f.read()

print(f"File size: {len(content)} bytes")

# ---- Step 1: pypdf ----
print("\n=== Step 1: pypdf ===")
from pypdf import PdfReader
reader = PdfReader(io.BytesIO(content))
all_text = []
for page in reader.pages:
    t = (page.extract_text() or "").strip()
    if t:
        all_text.append(t)
total_pypdf = sum(len(t) for t in all_text)
print(f"Pages with text: {len(all_text)}/{len(reader.pages)}, total chars: {total_pypdf}")

# ---- Step 2: pdfplumber ----
total_pl = 0
pl_text = ""
if total_pypdf < 100:
    print("\n=== Step 2: pdfplumber ===")
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pl_texts = []
            for page in pdf.pages:
                t = (page.extract_text() or "").strip()
                if t:
                    pl_texts.append(t)
            total_pl = sum(len(t) for t in pl_texts)
            pl_text = "\n\n".join(pl_texts)
            print(f"Pages with text: {len(pl_texts)}/{len(pdf.pages)}, total chars: {total_pl}")
    except Exception as e:
        print(f"pdfplumber error: {e}")

# ---- Step 3: Determine final text ----
if total_pypdf >= 100:
    final_text = "\n\n".join(all_text)
    method = "pypdf"
elif total_pl >= 100:
    final_text = pl_text
    method = "pdfplumber"
else:
    print("\n=== Step 3: Would attempt OCR (skipping in this test) ===")
    final_text = None
    method = "OCR (not tested)"

if final_text:
    print(f"\n=== Final text ({method}): {len(final_text)} chars ===")
    print(f"Preview: {repr(final_text[:200])}")

    from app.pipeline.splitter import split_text
    chunks = split_text(text=final_text, doc_type="faq")
    print(f"\n=== Chunks: {len(chunks)} ===")
    for c in chunks:
        print(f"  [{c.chunk_idx}] {len(c.text)} chars, heading='{c.heading_title}', level={c.heading_level}")
else:
    print("\n*** Would fall through to OCR ***")
    print("This PDF is image-based and needs OCR!")
