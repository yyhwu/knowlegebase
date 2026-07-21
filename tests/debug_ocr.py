"""测试 rag切分.pdf 的 OCR 提取"""
import os, sys, traceback
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

path = r"C:\Users\Administrator\Desktop\测试文件\rag切分.pdf"

print("Initializing PaddleOCR...")
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="ch", use_angle_cls=True)
print("OCR ready!")

import pypdfium2 as pdfium
import numpy as np

pdf = pdfium.PdfDocument(path)
print(f"Total pages: {len(pdf)}")

ocr_texts = []
for page_num in range(len(pdf)):
    try:
        page = pdf[page_num]
        bitmap = page.render(scale=2.0)
        pil_img = bitmap.to_pil()
        arr = np.array(pil_img.convert("RGB"))
        result = ocr.ocr(arr, cls=False)
        if result and result[0]:
            lines = [line[1][0] for line in result[0] if line]
            if lines:
                ocr_texts.append(f"[PDF 第{page_num+1}页 OCR]\n" + "\n".join(lines))
            print(f"Page {page_num+1}: {len(lines)} lines")
        else:
            print(f"Page {page_num+1}: 0 lines (result={result})")
    except Exception as e:
        print(f"Page {page_num+1} ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

pdf.close()

print(f"\n=== Summary ===")
print(f"Pages with OCR text: {len(ocr_texts)}")

if ocr_texts:
    full_text = "\n\n".join(ocr_texts)
    print(f"Total chars: {len(full_text)}")
    print(f"\nFirst 300 chars:")
    print(full_text[:300])
    print(f"\n... Last 200 chars:")
    print(full_text[-200:])

    # Test splitter
    from app.pipeline.splitter import split_text
    chunks = split_text(text=full_text, doc_type="faq")
    print(f"\n=== Chunks: {len(chunks)} ===")
    for c in chunks:
        print(f"  Chunk {c.chunk_idx}: {len(c.text)} chars, heading='{c.heading_title}'")
else:
    print("WARNING: No OCR text extracted at all!")
