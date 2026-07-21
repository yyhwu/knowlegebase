"""
/api/conversations — 会话管理
"""
import uuid
from fastapi import APIRouter, Query

from app.core.session import Conversation
from app.core.session_sqlite import get_session_store

router = APIRouter()


@router.post("/conversations")
async def api_create_conversation(
    title: str = "新会话",
    user_id: str = Query("default"),
):
    """显式创建新会话（前端"新会话"按钮调用）"""
    store = await get_session_store()
    conv = Conversation(
        id=str(uuid.uuid4()),
        title=title,
        user_id=user_id,
    )
    await store.create_conversation(conv)
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat(),
    }


@router.get("/conversations")
async def api_list_conversations(
    user_id: str = Query("default"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """获取会话列表"""
    store = await get_session_store()
    convs = await store.list_conversations(user_id=user_id, limit=limit, offset=offset)
    return {
        "total": len(convs),
        "conversations": [
            {
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in convs
        ],
    }


@router.get("/conversations/{conv_id}")
async def api_get_conversation(conv_id: str):
    """获取单个会话的完整消息"""
    store = await get_session_store()
    conv = await store.get_conversation(conv_id)
    if not conv:
        return {"error": "会话不存在"}, 404
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat(),
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "feedback": m.feedback,
                "latency_ms": m.latency_ms,
                "created_at": m.created_at.isoformat(),
            }
            for m in conv.messages
        ],
    }


@router.post("/conversations/{conv_id}/feedback")
async def api_update_feedback(conv_id: str, msg_id: str, feedback: int = Query(ge=-1, le=1)):
    """消息反馈：1=赞 -1=踩"""
    store = await get_session_store()
    ok = await store.update_feedback(msg_id, feedback)
    return {"ok": ok}


@router.delete("/conversations/{conv_id}")
async def api_delete_conversation(conv_id: str):
    """删除会话"""
    store = await get_session_store()
    ok = await store.delete_conversation(conv_id)
    return {"ok": ok}
