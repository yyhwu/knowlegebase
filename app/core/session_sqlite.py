"""
会话持久层 — SQLite 实现（异步）
接口设计允许未来替换为 PostgreSQL / Redis
"""
from __future__ import annotations

import aiosqlite
import json
import os
from datetime import datetime, timezone
from typing import Optional

from app.core.session import Conversation, Message, SessionStore

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sessions.db")


class SQLiteSessionStore(SessionStore):
    """SQLite 会话存储"""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    async def _get_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self._db_path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        return db

    async def init(self):
        """建表（应用启动时调用一次）"""
        db = await self._get_db()
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                user_id TEXT DEFAULT 'default',
                tenant_id TEXT DEFAULT 'default',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '[]',
                feedback INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);
        """)
        await db.commit()
        await db.close()

    async def create_conversation(self, conv: Conversation) -> Conversation:
        db = await self._get_db()
        await db.execute(
            "INSERT INTO conversations (id, title, user_id, tenant_id, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (conv.id, conv.title, conv.user_id, conv.tenant_id,
             conv.created_at.isoformat(), conv.updated_at.isoformat()),
        )
        await db.commit()
        await db.close()
        return conv

    async def add_message(self, msg: Message) -> Message:
        db = await self._get_db()
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources, feedback, latency_ms, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (msg.id, msg.conversation_id, msg.role, msg.content,
             json.dumps(msg.sources, ensure_ascii=False), msg.feedback,
             msg.latency_ms, msg.created_at.isoformat()),
        )
        await db.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), msg.conversation_id),
        )
        await db.commit()
        await db.close()
        return msg

    async def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        db = await self._get_db()
        async with db.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)) as cursor:
            row = await cursor.fetchone()
        if not row:
            await db.close()
            return None

        async with db.execute(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC",
            (conv_id,),
        ) as cursor:
            msg_rows = await cursor.fetchall()

        await db.close()

        messages = [
            Message(
                id=m["id"], conversation_id=m["conversation_id"],
                role=m["role"], content=m["content"],
                sources=json.loads(m["sources"]),
                feedback=m["feedback"], latency_ms=m["latency_ms"],
                created_at=datetime.fromisoformat(m["created_at"]),
            )
            for m in msg_rows
        ]

        return Conversation(
            id=row["id"], title=row["title"],
            user_id=row["user_id"], tenant_id=row["tenant_id"],
            messages=messages,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def list_conversations(
        self, user_id: str = "default", tenant_id: str = "default",
        limit: int = 20, offset: int = 0,
    ) -> list[Conversation]:
        db = await self._get_db()
        async with db.execute(
            "SELECT * FROM conversations WHERE user_id=? AND tenant_id=? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (user_id, tenant_id, limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()
        await db.close()
        return [
            Conversation(
                id=r["id"], title=r["title"],
                user_id=r["user_id"], tenant_id=r["tenant_id"],
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
            for r in rows
        ]

    async def update_feedback(self, msg_id: str, feedback: int) -> bool:
        db = await self._get_db()
        await db.execute("UPDATE messages SET feedback=? WHERE id=?", (feedback, msg_id))
        await db.commit()
        await db.close()
        return True

    async def delete_conversation(self, conv_id: str) -> bool:
        db = await self._get_db()
        await db.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
        await db.commit()
        await db.close()
        return True

    async def get_recent_messages(
        self, conv_id: str, limit: int = 10,
    ) -> list[dict[str, str]]:
        db = await self._get_db()
        async with db.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?",
            (conv_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        await db.close()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ── 全局单例 ──────────────────────────────

_store: Optional[SQLiteSessionStore] = None


async def get_session_store() -> SQLiteSessionStore:
    global _store
    if _store is None:
        _store = SQLiteSessionStore()
        await _store.init()
    return _store
