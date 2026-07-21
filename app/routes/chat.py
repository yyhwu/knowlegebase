"""
/api/chat — RAG 对话（企业版：会话持久化 + 结构化日志）
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.llm import get_llm
from app.core.logging import get_logger
from app.core.session import Conversation, Message
from app.core.session_sqlite import get_session_store
from app.models.schemas import ChatRequest, ChatResponse
from app.pipeline.guard import apply_guard
from app.pipeline.prompt import build_rag_prompt
from app.pipeline.retriever import search as rag_search

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    """
    RAG 对话接口（企业版）
    流程: 加载会话历史 → 检索 → 拼装 Prompt → LLM 生成 → Guard → 持久化
    """
    t_start = time.monotonic()
    store = await get_session_store()

    # ── 0. 会话管理 ──
    conv_id = req.conversation_id or str(uuid.uuid4())

    # 从服务端加载最近历史（取代前端传 history）
    recent = await store.get_recent_messages(conv_id, limit=10)
    history_dicts = [{"role": m["role"], "content": m["content"]} for m in recent]

    # 检查是否新建会话
    conv = await store.get_conversation(conv_id)
    is_new = conv is None
    if is_new:
        conv = Conversation(id=conv_id, title=req.query[:80], user_id=req.user_id or "default")
        await store.create_conversation(conv)

    # ── 保存用户消息 ──
    user_msg = Message(
        conversation_id=conv_id,
        role="user",
        content=req.query,
        created_at=datetime.now(timezone.utc),
    )
    await store.add_message(user_msg)

    # ── 1. 检索 ──
    docs = rag_search(
        query=req.query,
        top_k=10,
        doc_types=req.doc_types,
        department=req.department,
        history=history_dicts or None,
        expand_context=True,
    )

    # ── 2. 拼装 Prompt ──
    prompt = build_rag_prompt(
        query=req.query,
        docs=docs,
        history=history_dicts or None,
    )

    # ── 3. LLM 生成 ──
    llm = get_llm()
    raw_answer = llm.invoke(prompt)

    # ── 4. 后置校验 ──
    answer, warnings = apply_guard(raw_answer)

    # ── 5. 收集引用 ──
    source_ids = list({d.get("doc_id") for d in docs if d.get("doc_id")})

    # ── 6. 保存助手回复 ──
    latency = int((time.monotonic() - t_start) * 1000)
    assistant_msg = Message(
        conversation_id=conv_id,
        role="assistant",
        content=answer,
        sources=source_ids,
        latency_ms=latency,
        created_at=datetime.now(timezone.utc),
    )
    await store.add_message(assistant_msg)

    # ── 7. 结构化日志 ──
    logger.info("conv=%s query=\"%s\" sources=%d latency=%dms",
                conv_id[:8], req.query[:60], len(source_ids), latency)

    return ChatResponse(answer=answer, sources=source_ids)
