"""深入分析 splitter 对 OCR 文本的分割行为"""
import os, sys
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, r'c:\Users\Administrator\Desktop\knowleggeBase')

import re
from app.pipeline.splitter import split_text, split_structured, _has_md_headers

# 模拟两种 OCR 输出的格式
# 格式A: 有 [PDF 第N页 OCR] 标记 (ingest.py 中 ocr_texts 的格式)
ocr_text_A = """[PDF 第1页 OCR]
这是第一页OCR出来的第一行文字内容比较长需要测试切分效果。
这是第一页OCR出来的第二行文字内容比较长需要测试切分效果。
这是第一页OCR出来的第三行文字内容比较长需要测试切分效果。
这是第一页OCR出来的第四行文字内容比较长需要测试切分效果。
这是第一页OCR出来的第五行文字内容比较长需要测试切分效果。

[PDF 第2页 OCR]
这是第二页OCR出来的第一行文字内容比较长需要测试切分效果。
这是第二页OCR出来的第二行文字内容比较长需要测试切分效果。
这是第二页OCR出来的第三行文字内容比较长需要测试切分效果。
这是第二页OCR出来的第四行文字内容比较长需要测试切分效果。
这是第二页OCR出来的第五行文字内容比较长需要测试切分效果。"""

# 测试1: has_md_headers?
print(f"Test A - has_md_headers: {_has_md_headers(ocr_text_A)}")

# 测试2: split_flat 行为
chunks_A = split_text(ocr_text_A, "faq")
print(f"Test A - chunks: {len(chunks_A)}")
for c in chunks_A:
    print(f"  [{c.chunk_idx}] {len(c.text)} chars: {repr(c.text[:60])}")

# 测试3: 模拟只有一页的短文本
ocr_text_short = """[PDF 第1页 OCR]
短文本测试。"""

chunks_short = split_text(ocr_text_short, "faq")
print(f"\nTest Short - chunks: {len(chunks_short)}, text len: {len(ocr_text_short)}")

# 测试4: 模拟非常长的单块文本（无段落分隔）
ocr_text_long = "这是一个非常长的句子。" * 80  # 800 chars, no \n\n separator
print(f"\nTest Long - text len: {len(ocr_text_long)}")
chunks_long = split_text(ocr_text_long, "faq")
print(f"Test Long - chunks: {len(chunks_long)}")
for c in chunks_long[:3]:
    print(f"  [{c.chunk_idx}] {len(c.text)} chars: {repr(c.text[:60])}")

# 测试5: 关键 - 模拟仅一页且非常长的文本
ocr_text_one_page = "ABC" * 300  # 900 chars
print(f"\nTest OnePage - text len: {len(ocr_text_one_page)}")
chunks_onepage = split_text(ocr_text_one_page, "faq")
print(f"Test OnePage - chunks: {len(chunks_onepage)}")
for c in chunks_onepage[:4]:
    print(f"  [{c.chunk_idx}] {len(c.text)} chars: {repr(c.text[:60])}")
