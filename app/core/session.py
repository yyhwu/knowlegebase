"""
会话持久层抽象 — 可替换存储后端（SQLite / PostgreSQL / Redis）
企业版：接口优先设计，支持未来平滑迁移
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


# ── 数据模型 ──────────────────────────────

@dataclass
class Message:
    """单条消息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = ""
    role: str = "user"          # user | assistant
    content: str = ""
    sources: list[str] = field(default_factory=list)  # doc_id 列表
    feedback: int = 0           # 1:赞 -1:踩 0:无
    latency_ms: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Conversation:
    """一个会话"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""             # 首条 query 截取
    user_id: str = "default"
    tenant_id: str = "default"
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── 抽象接口 ──────────────────────────────

class SessionStore(ABC):
    """会话存储抽象（可替换 PostgreSQL / Redis / MongoDB 等）"""

    @abstractmethod
    async def create_conversation(self, conv: Conversation) -> Conversation:
        """新建会话"""
        ...

    @abstractmethod
    async def add_message(self, msg: Message) -> Message:
        """追加消息"""
        ...

    @abstractmethod
    async def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """获取会话（含所有消息）"""
        ...

    @abstractmethod
    async def list_conversations(
        self, user_id: str = "default", tenant_id: str = "default",
        limit: int = 20, offset: int = 0,
    ) -> list[Conversation]:
        """列出用户的会话列表"""
        ...

    @abstractmethod
    async def update_feedback(self, msg_id: str, feedback: int) -> bool:
        """更新消息反馈"""
        ...

    @abstractmethod
    async def delete_conversation(self, conv_id: str) -> bool:
        """删除会话"""
        ...

    @abstractmethod
    async def get_recent_messages(
        self, conv_id: str, limit: int = 10,
    ) -> list[dict[str, str]]:
        """获取最近 N 轮消息（供 prompt 拼装用）"""
        ...
