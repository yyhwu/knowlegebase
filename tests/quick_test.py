import os, sys, io, tempfile
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, r"c:\Users\Administrator\Desktop\knowleggeBase")

for fname in ["rag切分.pdf", "rag优化.pdf"]:
    path = rf"C:\Users\Administrator\Desktop\测试文件\{fname}"
    print(f"\n{'='*60}")
    print(f"FILE: {fname}")
    
    with open(path, "rb") as f:
        content = f.read()
    
    # Exact same flow as _extract_pdf_fallback
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    all_text = []
    for page in reader.pages:
        t = (page.extract_text() or "").strip()
        if t: all_text.append(t)
    t_pypdf = sum(len(t) for t in all_text)
    
    import pdfplumber
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        all_text_pl = []
        for page in pdf.pages:
            t = (page.extract_text() or "").strip()
            if t: all_text_pl.append(t)
    t_pl = sum(len(t) for t in all_text_pl)
    
    print(f"pypdf: {t_pypdf} chars, pdfplumber: {t_pl} chars")
    
    if t_pypdf >= 100:
        final = "\n\n".join(all_text)
        method = "pypdf"
    elif t_pl >= 100:
        final = "\n\n".join(all_text_pl)
        method = "pdfplumber"
    else:
        print("  -> Would go to OCR (skipping for speed)")
        continue
    
    print(f"  Method: {method}, text: {len(final)} chars")
    
    from app.pipeline.splitter import split_text
    chunks = split_text(text=final, doc_type="faq")
    print(f"  Chunks: {len(chunks)}")
