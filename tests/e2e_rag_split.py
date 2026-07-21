"""完整端到端测试 - rag切分.pdf"""
import os, io, sys, tempfile
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, r"c:\Users\Administrator\Desktop\knowleggeBase")

path = r"C:\Users\Administrator\Desktop\测试文件\rag切分.pdf"
with open(path, "rb") as f:
    content = f.read()
print(f"File: {len(content)} bytes")

# Step 1: pypdf
from pypdf import PdfReader
reader = PdfReader(io.BytesIO(content))
at = [(p.extract_text() or "").strip() for p in reader.pages]
tp = sum(len(t) for t in at)
print(f"Step1 pypdf: {tp} chars, {sum(1 for t in at if t)}/{len(at)} pages")

# Step 2: pdfplumber
import pdfplumber
with pdfplumber.open(io.BytesIO(content)) as pdf:
    at2 = [(p.extract_text() or "").strip() for p in pdf.pages]
tp2 = sum(len(t) for t in at2)
print(f"Step2 pdfplumber: {tp2} chars, {sum(1 for t in at2 if t)}/{len(at2)} pages")

# Step 3: OCR
if tp < 100 and tp2 < 100:
    print("Step3: OCR needed!")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="ch", use_angle_cls=True)
    print("  OCR ready")
    import pypdfium2 as pdfium, numpy as np
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tpth = tmp.name
    
    pdf = pdfium.PdfDocument(tpth)
    print(f"  Pages: {len(pdf)}")
    ot = []
    for pn in range(len(pdf)):
        page = pdf[pn]
        bmp = page.render(scale=2.0)
        arr = np.array(bmp.to_pil().convert("RGB"))
        res = ocr.ocr(arr, cls=False)
        if res and res[0]:
            lines = [l[1][0] for l in res[0] if l]
            if lines:
                ot.append(f"[PDF 第{pn+1}页 OCR]\n" + "\n".join(lines))
                print(f"  Page {pn+1}: {len(lines)} lines")
    pdf.close()
    os.unlink(tpth)
    
    text = "\n\n".join(ot)
    print(f"  Total OCR text: {len(text)} chars, {len(ot)} pages with text")
else:
    text = "\n\n".join(at) if tp >= 100 else "\n\n".join(at2)
    print(f"Text from {'pypdf' if tp>=100 else 'pdfplumber'}: {len(text)} chars")

# Step 4: Split
from app.pipeline.splitter import split_text, _has_md_headers
print(f"\nhas_md_headers: {_has_md_headers(text)}")
chunks = split_text(text, "faq")
print(f"Chunks: {len(chunks)}")
for c in chunks[:5]:
    print(f"  [{c.chunk_idx}] {len(c.text)} chars: {c.text[:80]}...")
if len(chunks) == 1:
    print(f"\n!!! ONLY 1 CHUNK !!! Text length: {len(text)}")
