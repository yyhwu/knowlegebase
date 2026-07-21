"""
文档分割器 — 标题结构感知 + 小chunk + 父文档上下文
企业 RAG 版：按文档结构层级切分，保留父子关系用于检索时上下文扩展
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# ── 数据结构 ──────────────────────────────

@dataclass
class Chunk:
    """一个文档片段（小 chunk，入库 Milvus）"""
    text: str
    doc_id: str
    chunk_idx: int
    doc_type: str
    # ── 层级信息（替代旧的 FACT/STYLE 分类）──
    parent_idx: int | None = None   # 父 chunk 索引，检索时用于上下文扩展
    heading_level: int = 0          # 所属标题层级: 0=无标题, 1=h1, 2=h2 ...
    heading_title: str = ""         # 所属标题文本

    metadata: dict = field(default_factory=dict)
    md5: str = ""

    def __post_init__(self):
        if not self.md5:
            self.md5 = hashlib.md5(self.text.encode("utf-8")).hexdigest()


@dataclass
class ParentChunk:
    """父 chunk（章节级完整上下文，不入库 Milvus，仅用于检索时扩展）"""
    text: str
    chunk_idx: int
    heading_level: int
    heading_title: str


# ── 标题结构分割 ──────────────────────────

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

# 中文段落分隔符
PARAGRAPH_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]


def _has_md_headers(text: str) -> bool:
    return bool(re.search(r'^#{1,4}\s', text, re.MULTILINE))


def _is_table_line(text: str) -> bool:
    return bool(re.match(r'^\|.+\|$', text.strip()))


def _extract_heading(section) -> tuple[int, str]:
    """从 markdown section 提取最深标题层级和文本"""
    headings = section.metadata if hasattr(section, 'metadata') else {}
    # 从深到浅找最深层级
    for level_name in ("h4", "h3", "h2", "h1"):
        if headings.get(level_name):
            return int(level_name[1]), headings[level_name]
    return 0, ""


# ── 主入口 ────────────────────────────────

def split_structured(
    text: str,
    doc_type: str,
    doc_id: str | None = None,
    metadata: dict | None = None,
    child_chunk_size: int = 256,
    child_chunk_overlap: int = 32,
) -> tuple[list[Chunk], list[ParentChunk]]:
    """
    结构感知分割（核心）：
      1. 有 Markdown 标题 → 按 h1~h4 切分为 sections
      2. 每个 section = 一个 ParentChunk（章节级完整上下文）
      3. 每个 section 再切分为小 Chunk（256 token 级）
      4. 表格行保持完整不切割
      5. 无标题文档 → 整篇为父chunk，按段落切小

    返回: (入库的小chunks, 检索时用于扩展上下文的父chunks)
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())
    meta = metadata or {}

    if _has_md_headers(text):
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
        ).split_text(text)
        return _split_sections(sections, doc_id, doc_type, meta, child_chunk_size, child_chunk_overlap)
    else:
        return _split_flat(text, doc_id, doc_type, meta, child_chunk_size, child_chunk_overlap)


def _split_sections(sections, doc_id, doc_type, metadata, child_size, child_overlap):
    """按章节结构切分"""
    small_chunks: list[Chunk] = []
    parent_chunks: list[ParentChunk] = []

    for sec_idx, section in enumerate(sections):
        sec_text = section.page_content if hasattr(section, 'page_content') else str(section)
        if not sec_text.strip():
            continue

        level, title = _extract_heading(section)
        parent_chunks.append(ParentChunk(text=sec_text, chunk_idx=sec_idx, heading_level=level, heading_title=title))

        # 表格行 → 保持完整
        if _is_table_line(sec_text):
            small_chunks.append(Chunk(text=sec_text.strip(), doc_id=doc_id, chunk_idx=len(small_chunks),
                                      doc_type=doc_type, parent_idx=sec_idx, heading_level=level,
                                      heading_title=title, metadata=metadata))
            continue

        # 章节内小 chunk 切分
        sub_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=child_overlap,
                                                       separators=PARAGRAPH_SEPARATORS)
        for sub_text in sub_splitter.split_text(sec_text):
            if sub_text.strip():
                small_chunks.append(Chunk(text=sub_text.strip(), doc_id=doc_id, chunk_idx=len(small_chunks),
                                          doc_type=doc_type, parent_idx=sec_idx, heading_level=level,
                                          heading_title=title, metadata=metadata))

    return small_chunks, parent_chunks


def _split_flat(text, doc_id, doc_type, metadata, child_size, child_overlap):
    """无标题文档：整篇为父chunk"""
    parent = ParentChunk(text=text, chunk_idx=0, heading_level=0, heading_title="")
    splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=child_overlap,
                                               separators=PARAGRAPH_SEPARATORS)
    chunks: list[Chunk] = []
    for i, t in enumerate(splitter.split_text(text)):
        if t.strip():
            chunks.append(Chunk(text=t.strip(), doc_id=doc_id, chunk_idx=i, doc_type=doc_type,
                                parent_idx=0, heading_level=0, heading_title="", metadata=metadata))
    return chunks, [parent]


# ── 兼容旧接口 ────────────────────────────

def split_text(text: str, doc_type: str, doc_id: str | None = None, metadata: dict | None = None) -> list[Chunk]:
    """兼容旧 API：只返回小 chunk（供 ingest.py 过渡使用）"""
    chunks, __ = split_structured(text=text, doc_type=doc_type, doc_id=doc_id, metadata=metadata)
    return chunks
