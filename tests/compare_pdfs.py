"""对比两个 PDF"""
import os, sys, io
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

for fname in ["rag切分.pdf", "rag优化.pdf"]:
    path = rf"C:\Users\Administrator\Desktop\测试文件\{fname}"
    print(f"\n{'='*60}")
    print(f"FILE: {fname} ({os.path.getsize(path)/1024:.0f} KB)")
    
    with open(path, 'rb') as f:
        content = f.read()
    
    # pypdf
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    texts = [(page.extract_text() or "").strip() for page in reader.pages]
    total = sum(len(t) for t in texts)
    pages_with = sum(1 for t in texts if t)
    print(f"pypdf: {pages_with}/{len(reader.pages)} pages, {total} chars")
    
    # pdfplumber
    if total < 100:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pl_texts = [(page.extract_text() or "").strip() for page in pdf.pages]
            pl_total = sum(len(t) for t in pl_texts)
            pl_pages = sum(1 for t in pl_texts if t)
            print(f"pdfplumber: {pl_pages}/{len(pdf.pages)} pages, {pl_total} chars")
    
    print(f"Result: {'TEXT extracted' if total >= 100 else 'IMAGE-only PDF -> needs OCR'}")
