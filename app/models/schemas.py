"""
Pydantic 请求 / 响应模型
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── 检索请求 / 响应 ──────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=50)
    doc_types: list[str] | None = None   # 按类型过滤
    department: str | None = None          # 按部门过滤


class SearchResult(BaseModel):
    doc_id: str
    chunk_idx: int
    text: str
    score: float
    doc_type: str
    metadata: dict | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int


# ── 对话请求 / 响应 ──────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None   # 会话ID（null则新建）
    user_id: str | None = "default"       # 用户标识
    history: list[ChatMessage] | None = None  # 前端历史（可选，优先用服务端存储）
    doc_types: list[str] | None = None
    department: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)  # 引用的 doc_id 列表


# ── 文档入库请求 / 响应 ──────────────────

class IngestMetadata(BaseModel):
    """入库时附带的扩展元数据"""
    doc_type: str = Field(..., description="sop | spec | faq | contract | script | case")
    priority: str = Field(default="medium", pattern="^(high|medium|low)$")
    source: str = ""           # 来源文件名
    department: str = ""        # 部门
    tags: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    message: str = "入库成功"


# ── 索引管理 ─────────────────────────────

class ReindexResponse(BaseModel):
    message: str
    collection: str
