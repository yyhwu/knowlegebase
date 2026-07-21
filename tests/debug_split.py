"""调试 PDF 分片差异"""
import os, sys
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

base = r"C:\Users\Administrator\Desktop\测试文件"

for fname in ["rag切分.pdf", "rag优化.pdf"]:
    path = os.path.join(base, fname)
    print(f"\n{'='*60}")
    print(f"FILE: {fname}  ({os.path.getsize(path)/1024:.0f} KB)")
    print(f"{'='*60}")

    # 1. pypdf
    print("\n--- pypdf ---")
    from pypdf import PdfReader
    reader = PdfReader(path)
    print(f"Pages: {len(reader.pages)}")
    all_text = []
    for page in reader.pages:
        t = (page.extract_text() or "").strip()
        if t:
            all_text.append(t)
    total = sum(len(t) for t in all_text)
    print(f"Pages with text: {len(all_text)}, total chars: {total}")
    for i, t in enumerate(all_text[:3]):
        print(f"  Page {i+1} ({len(t)} chars): {repr(t[:80])}")
    if len(all_text) > 3:
        print(f"  ... ({len(all_text)-3} more pages)")

    # 2. pdfplumber
    print("\n--- pdfplumber ---")
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            print(f"Pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages[:3]):
                t = (page.extract_text() or "").strip()
                print(f"  Page {i+1} ({len(t)} chars): {repr(t[:80]) if t else '(empty)'}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. 模拟 splitter 行为
    print("\n--- splitter ---")
    # 提取最终文本（用 ingest 的逻辑：pypdf优先）
    text = None
    if total >= 100:
        text = "\n\n".join(all_text)
    else:
        # fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pl_texts = []
                for page in pdf.pages:
                    t = (page.extract_text() or "").strip()
                    if t:
                        pl_texts.append(t)
                if sum(len(t) for t in pl_texts) >= 100:
                    text = "\n\n".join(pl_texts)
        except:
            pass

    if text:
        print(f"Total extracted text: {len(text)} chars")
        print(f"Text preview (first 200 chars): {repr(text[:200])}")

        from app.pipeline.splitter import split_text
        chunks = split_text(text=text, doc_type="faq")
        print(f"Chunks produced: {len(chunks)}")
        for c in chunks[:3]:
            print(f"  Chunk {c.chunk_idx}: {len(c.text)} chars, heading='{c.heading_title}', level={c.heading_level}")
            print(f"    text: {repr(c.text[:80])}")
        if len(chunks) > 3:
            print(f"  ... ({len(chunks)-3} more chunks)")
    else:
        print("WARNING: No text extracted! Would fall through to OCR.")
