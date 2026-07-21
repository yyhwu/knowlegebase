"""完整模拟 ingest 流程 - 测试 rag切分.pdf"""
import os, sys, io, tempfile, traceback
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, r"c:\Users\Administrator\Desktop\knowleggeBase")

# 1. 模拟上传：读取文件
path = r"C:\Users\Administrator\Desktop\测试文件\rag切分.pdf"
with open(path, "rb") as f:
    content = f.read()
print(f"1. File read: {len(content)} bytes")

# 2. 模拟 pypdf 提取
from pypdf import PdfReader
reader = PdfReader(io.BytesIO(content))
all_text_pypdf = []
for page in reader.pages:
    t = (page.extract_text() or "").strip()
    if t:
        all_text_pypdf.append(t)
total_pypdf = sum(len(t) for t in all_text_pypdf)
print(f"2. pypdf: {len(all_text_pypdf)}/{len(reader.pages)} pages, {total_pypdf} chars")

# 3. 模拟 pdfplumber 提取
total_pl = 0
if total_pypdf < 100:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        all_text_pl = []
        for page in pdf.pages:
            t = (page.extract_text() or "").strip()
            if t:
                all_text_pl.append(t)
        total_pl = sum(len(t) for t in all_text_pl)
    print(f"3. pdfplumber: {len(all_text_pl)}/{len(pdf.pages)} pages, {total_pl} chars")

# 4. 模拟 OCR 提取
text = None
if total_pypdf < 100 and total_pl < 100:
    print("4. Need OCR! Initializing PaddleOCR...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="ch", use_angle_cls=True)
    print("   OCR ready!")
    
    import pypdfium2 as pdfium
    import numpy as np
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    print(f"   Temp file: {tmp_path}")
    
    try:
        pdf = pdfium.PdfDocument(tmp_path)
        print(f"   Pages: {len(pdf)}")
        ocr_texts = []
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            bitmap = page.render(scale=2.0)
            pil_img = bitmap.to_pil()
            arr = np.array(pil_img.convert("RGB"))
            result = ocr.ocr(arr, cls=False)
            if result and result[0]:
                lines = [line[1][0] for line in result[0] if line]
                if lines:
                    ocr_texts.append(f"[PDF 第{page_num+1}页 OCR]\n" + "\n".join(lines))
        pdf.close()
        if ocr_texts:
            text = "\n\n".join(ocr_texts)
        print(f"   OCR result: {len(ocr_texts)} pages with text, {len(text) if text else 0} total chars")
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
else:
    if total_pypdf >= 100:
        text = "\n\n".join(all_text_pypdf)
        print(f"4. Using pypdf text: {len(text)} chars")
    else:
        text = "\n\n".join(all_text_pl)
        print(f"4. Using pdfplumber text: {len(text)} chars")

if text is None:
    print("ERROR: No text extracted!")
else:
    print(f"\n5. Extracted text: {len(text)} chars")
    print(f"   First 150 chars: {repr(text[:150])}")
    print(f"   Has MD headers: {bool(__import__('re').search(r'^#{1,4}\s', text, __import__('re').MULTILINE))}")
    
    # 6. 分块
    from app.pipeline.splitter import split_text
    chunks = split_text(text=text, doc_type="faq")
    print(f"\n6. Chunks produced: {len(chunks)}")
    for c in chunks[:5]:
        print(f"   [{c.chunk_idx}] {len(c.text)} chars: {repr(c.text[:80])}")
    if len(chunks) > 5:
        print(f"   ... ({len(chunks)-5} more)")
