"""
/api/ingest — 文档入库
支持：TXT / MD / PDF（含 OCR 图片文字提取）/ DOCX
"""
import io
import os
import uuid
from typing import Any

# PaddlePaddle 2.x + pymilvus 3.x protobuf 冲突修复
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.core.embeddings import get_embedder
from app.core.logging import get_logger
from app.core.milvus import get_milvus_collection
from app.models.schemas import IngestResponse
from app.pipeline.splitter import split_text

logger = get_logger(__name__)

router = APIRouter()

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx", ".json", ".yaml", ".yml"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_FILE_COUNT = 50               # 最多 50 个文档

# ── OCR 惰性单例 ──────────────────────────
_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        try:
            from paddleocr import PaddleOCR
            _ocr = PaddleOCR(lang="ch", use_angle_cls=True)
            logger.info("PaddleOCR 初始化成功")
        except Exception as e:
            logger.exception("PaddleOCR 初始化失败: %s: %s", type(e).__name__, e)
            _ocr = False
    return _ocr
    return _ocr


def _extract_text(file: UploadFile) -> str:
    """根据文件类型提取文本（参考 pdf_reading.md 推荐优先级的降级链路）"""
    ext = os.path.splitext(file.filename or "")[-1].lower()

    if ext in {".txt", ".md", ".csv", ".json", ".yaml", ".yml"}:
        return file.file.read().decode("utf-8", errors="replace")

    if ext == ".pdf":
        return _extract_pdf_fallback(file)

    if ext == ".docx":
        try:
            import docx2txt
            return docx2txt.process(file.file)
        except ImportError:
            raise HTTPException(500, "docx2txt 未安装，无法解析 DOCX")

    if ext == ".xlsx":
        return _extract_excel(file)

    raise HTTPException(400, f"不支持的文件格式: {ext}")


def _extract_pdf_fallback(file: UploadFile) -> str:
    """
    PDF 降级提取链路（参考 pdf_reading.md 推荐优先级）:
      1. pypdf extract_text() — 最快，适合电子文档
      2. pdfplumber — 保留布局 + 提取表格
      3. PaddleOCR — 扫描件/图片型 PDF
    """
    content = file.file.read()
    file.file = io.BytesIO(content)

    # ── 1. pypdf ──
    total_len_pypdf = 0  # 默认值，防止后续 NameError
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        all_text = []
        for page in reader.pages:
            t = (page.extract_text() or "").strip()
            if t:
                all_text.append(t)
        total_len_pypdf = sum(len(t) for t in all_text)
        logger.info("pypdf 提取: %d 页有文字, 总字数 %d", len(all_text), total_len_pypdf)

        if total_len_pypdf >= 100:  # 不少于 100 字才当有效提取
            return "\n\n".join(all_text)
    except Exception as e:
        logger.warning("pypdf 异常: %s", e)

    # ── 2. pdfplumber（保留布局 + 表格）──
    total_len_pl = 0  # 默认值，防止后续 NameError
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            all_text = []
            for page in pdf.pages:
                t = (page.extract_text() or "").strip()
                if t:
                    all_text.append(t)
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        rows = [" | ".join(str(c or "") for c in row) for row in table]
                        all_text.append("[表格]\n" + "\n".join(rows))
            total_len_pl = sum(len(t) for t in all_text)
            logger.info("pdfplumber 提取: %d 个片段, 总字数 %d", len(all_text), total_len_pl)

            if total_len_pl >= 100:  # 不少于 100 字才当有效提取
                return "\n\n".join(all_text)
    except Exception as e:
        logger.warning("pdfplumber 异常: %s", e)

    # ── 3. pypdfium2 渲染 + PaddleOCR（扫描件/图片型）──
    ocr = _get_ocr()
    logger.info("OCR 状态: %s (pypdf=%d 字, pdfplumber=%d 字)",
                '可用' if ocr else '不可用', total_len_pypdf, total_len_pl)
    if ocr:
        try:
            import pypdfium2 as pdfium
            import numpy as np

            # pdfium 需要文件路径，不能传 bytes
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                pdf = pdfium.PdfDocument(tmp_path)
                n_pages = len(pdf)
                logger.info("OCR 共 %d 页，开始逐页识别...", n_pages)
                ocr_texts = []
                for page_num in range(n_pages):
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
                            logger.debug("第 %d/%d 页: %d 行", page_num + 1, n_pages, len(lines))
                        else:
                            logger.debug("第 %d/%d 页: 0 行 (空白页?)", page_num + 1, n_pages)
                    except Exception as page_err:
                        logger.warning("第 %d/%d 页失败: %s", page_num + 1, n_pages, page_err)
                pdf.close()
                logger.info("OCR 完成: %d/%d 页有文字, 总计约 %d 字符",
                           len(ocr_texts), n_pages, sum(len(t) for t in ocr_texts))
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            if ocr_texts:
                result_text = "\n\n".join(ocr_texts)
                logger.info("OCR 最终文本长度: %d 字符", len(result_text))
                return result_text
            else:
                logger.warning("OCR 未识别到任何文字（所有页空白或识别失败）")
        except Exception:
            logger.exception("OCR 整体异常")
    else:
        logger.warning("OCR 不可用，跳过 OCR 路径")

    raise HTTPException(
        400,
        "PDF 无法提取文字。该文件可能为纯图片扫描件、加密文档或损坏文件。"
        "请确认：1) 非加密PDF 2) PaddleOCR已加载（首次约30秒） 3) 文件未损坏"
    )


def _extract_excel(file: UploadFile) -> str:
    """Excel 提取（参考 excel_reading.md）"""
    try:
        import pandas as pd
        content = file.file.read()

        # 读取所有 sheet
        xl = pd.ExcelFile(io.BytesIO(content))
        all_parts = []

        for sheet_name in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet_name)
            if df.empty:
                continue
            # 只读前 500 行防止超大文件
            df = df.head(500)
            all_parts.append(f"[Sheet: {sheet_name}]\n" + df.to_string(index=False))

        if not all_parts:
            raise HTTPException(400, "Excel 文件无有效数据")

        return "\n\n".join(all_parts)
    except ImportError:
        raise HTTPException(500, "pandas/openpyxl 未安装，无法解析 Excel")
    except Exception as e:
        raise HTTPException(400, f"Excel 解析失败: {str(e)}")


@router.post("/ingest", response_model=IngestResponse)
async def api_ingest(
    file: UploadFile = File(..., description="上传文档"),
    doc_type: str = Form(..., description="sop | spec | faq | contract | script | case"),
    priority: str = Form("medium"),
    source: str = Form(""),
    department: str = Form(""),
    tags: str = Form(""),  # 逗号分隔
):
    """文档入库：上传 → 解析 → 分块 → 向量化 → 写入 Milvus"""

    # 校验文件类型
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式: {ext}。支持: {SUPPORTED_EXTENSIONS}")

    # 读取内容
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"文件过大，最大支持 {MAX_FILE_SIZE // 1024 // 1024}MB")

    # 检查同名文件 + 数量限制
    client = get_milvus_collection()
    try:
        all_docs = client.query(
            collection_name=settings.collection_name,
            filter="chunk_idx == 0",
            output_fields=["doc_id", "metadata"],
            limit=MAX_FILE_COUNT + 1,
        )
    except Exception:
        all_docs = []  # 空库时查询可能出错

    # 去重：按 metadata.source 检查
    for d in all_docs:
        src = (d.get("metadata") or {}).get("source", "")
        if src and src == (file.filename or ""):
            raise HTTPException(400, f"文件 '{file.filename}' 已存在，请勿重复上传")

    # 数量限制
    unique_docs = len({d["doc_id"] for d in all_docs})
    if unique_docs >= MAX_FILE_COUNT:
        raise HTTPException(400, f"已达到文档数量上限 {MAX_FILE_COUNT}，请先清理后再上传")

    file.file = io.BytesIO(content)
    text = _extract_text(file)

    if not text or not text.strip():
        raise HTTPException(400, "文件内容为空，无法入库")

    # 分块
    doc_id = str(uuid.uuid4())
    metadata: dict[str, Any] = {
        "priority": priority,
        "source": source or (file.filename or "unknown"),
        "department": department,
        "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
    }

    chunks = split_text(
        text=text,
        doc_type=doc_type,
        doc_id=doc_id,
        metadata=metadata,
    )

    if not chunks:
        raise HTTPException(400, "文档分块后无有效内容")

    if len(chunks) == 1 and len(text) < 500:
        logger.warning("文档 %s: 提取 %d 字符, 切分为 %d 个分片 (可能文本提取异常)",
                      file.filename, len(text), len(chunks))
    else:
        logger.info("文档 %s: 提取 %d 字符, 切分为 %d 个分片", file.filename, len(text), len(chunks))

    # 向量化 + 写入 (pymilvus 3.x)
    embedder = get_embedder()
    client = get_milvus_collection()  # MilvusClient

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(texts)

    # pymilvus 3.x: insert 接受 list[dict]
    rows = [
        {
            "doc_id": c.doc_id,
            "chunk_idx": c.chunk_idx,
            "text": c.text,
            "embedding": emb,
            "doc_type": c.doc_type,
            "parent_idx": c.parent_idx or -1,
            "heading_title": c.heading_title or "",
            "heading_level": c.heading_level,
            "metadata": c.metadata,
            "sparse_vector": {},
        }
        for c, emb in zip(chunks, embeddings)
    ]

    client.insert(collection_name=settings.collection_name, data=rows)

    return IngestResponse(
        doc_id=doc_id,
        chunks=len(chunks),
        message="入库成功",
    )
