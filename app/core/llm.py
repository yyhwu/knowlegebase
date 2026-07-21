"""
LLM 管理（DashScope 为主，预留本地模型接口）
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.language_models import BaseChatModel

from app.config import settings


# ── 抽象接口 ──────────────────────────────

class BaseLLM(ABC):
    """LLM 统一接口"""

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        ...

    @abstractmethod
    def get_langchain_model(self) -> BaseChatModel:
        ...


# ── DashScope 实现 ───────────────────────

class DashScopeLLM(BaseLLM):
    """阿里云通义千问"""

    def __init__(self) -> None:
        self._model = ChatTongyi(
            model=settings.llm_model_name,
            dashscope_api_key=settings.dashscope_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    def invoke(self, prompt: str) -> str:
        response = self._model.invoke(prompt)
        return response.content  # type: ignore[return-value]

    def get_langchain_model(self) -> BaseChatModel:
        return self._model


# ── 本地模型桩（预留扩展）─────────────────

class LocalLLM(BaseLLM):
    """本地模型接口（待实现）"""
    # TODO: 集成 vLLM / ollama
    pass


# ── 惰性单例 ──────────────────────────────

_llm: BaseLLM | None = None


def get_llm() -> BaseLLM:
    global _llm
    if _llm is None:
        if settings.llm_provider == "dashscope":
            _llm = DashScopeLLM()
        elif settings.llm_provider == "local":
            _llm = LocalLLM()
        else:
            raise ValueError(f"不支持的 LLM provider: {settings.llm_provider}")
    return _llm
