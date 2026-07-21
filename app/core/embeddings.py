"""
Embedding 模型管理（DashScope 为主，预留本地模型接口）
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_community.embeddings import DashScopeEmbeddings

from app.config import settings


# ── 抽象接口 ──────────────────────────────

class BaseEmbedder(ABC):
    """Embedding 模型统一接口"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


# ── DashScope 实现 ───────────────────────

class DashScopeEmbedder(BaseEmbedder):
    """阿里云 DashScope text-embedding-v4"""

    def __init__(self) -> None:
        self._model = DashScopeEmbeddings(
            model=settings.embedding_model_name,
            dashscope_api_key=settings.dashscope_api_key,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)


# ── 本地模型桩（预留扩展）─────────────────

class LocalEmbedder(BaseEmbedder):
    """本地模型接口（待实现）"""
    # TODO: 集成 sentence-transformers 或 huggingface
    pass


# ── 惰性单例 ──────────────────────────────

_embedder: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder is None:
        if settings.embedding_provider == "dashscope":
            _embedder = DashScopeEmbedder()
        elif settings.embedding_provider == "local":
            _embedder = LocalEmbedder()
        else:
            raise ValueError(f"不支持的 embedding provider: {settings.embedding_provider}")
    return _embedder
