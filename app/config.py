"""
配置中心 — 加载 .env + YAML，暴露统一 Settings 单例
"""
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings


# ── 项目根目录 ──────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    """加载 YAML 文件，解析 ${ENV:default} 占位符"""
    raw = path.read_text(encoding="utf-8")

    # 替换 ${VAR:default} 或 ${VAR}
    import re

    def _replace_env(m: re.Match) -> str:
        var = m.group(1)
        default = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
        return os.getenv(var, default)

    resolved = re.sub(r"\$\{(\w+)(?::([^}]*))?\}", _replace_env, raw)
    return yaml.safe_load(resolved)


# ── 加载配置 ──────────────────────────────
_yaml_config = _load_yaml(ROOT_DIR / "config" / "settings.yml")


class Settings(BaseSettings):
    """应用配置"""

    # DashScope
    dashscope_api_key: str = ""

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    collection_name: str = "knowledge_base"

    # Embedding
    embedding_provider: str = "dashscope"
    embedding_model_name: str = "text-embedding-v4"
    embedding_dimension: int = 1024

    # LLM
    llm_provider: str = "dashscope"
    llm_model_name: str = "qwen-plus"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 8192

    # 分块
    fact_chunk_size: int = 768
    fact_chunk_overlap: int = 128
    style_chunk_size: int = 256
    style_chunk_overlap: int = 32

    # 检索
    fact_top_k: int = 6
    style_top_k: int = 2

    # Reranker
    reranker_provider: str = "dashscope"
    reranker_model_name: str = "gte-rerank"
    reranker_top_n: int = 5

    # 缓存
    cache_embedding_ttl: int = 600
    cache_search_ttl: int = 120

    # Guard
    guard_max_answer_length: int = 8000
    guard_forbidden_words: list[str] = ["我保证", "绝对没问题"]

    # 应用
    app_env: str = "development"
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}

    @classmethod
    def load(cls) -> "Settings":
        """从 YAML + .env 合并加载"""
        s = cls()

        # ── Milvus ──
        mc = _yaml_config.get("milvus", {})
        s.milvus_host = os.getenv("MILVUS_HOST", mc.get("host", s.milvus_host))
        s.milvus_port = int(os.getenv("MILVUS_PORT", mc.get("port", s.milvus_port)))
        s.collection_name = mc.get("collection_name", s.collection_name)

        # ── Embedding ──
        ec = _yaml_config.get("embedding", {})
        s.embedding_provider = ec.get("provider", s.embedding_provider)
        s.embedding_model_name = ec.get("model_name", s.embedding_model_name)
        s.embedding_dimension = ec.get("dimension", s.embedding_dimension)

        # ── LLM ──
        lc = _yaml_config.get("llm", {})
        s.llm_provider = lc.get("provider", s.llm_provider)
        s.llm_model_name = lc.get("model_name", s.llm_model_name)
        s.llm_temperature = lc.get("temperature", s.llm_temperature)
        s.llm_max_tokens = lc.get("max_tokens", s.llm_max_tokens)

        # ── Chunking ──
        cc = _yaml_config.get("chunking", {})
        s.fact_chunk_size = cc.get("fact", {}).get("chunk_size", s.fact_chunk_size)
        s.fact_chunk_overlap = cc.get("fact", {}).get(
            "chunk_overlap", s.fact_chunk_overlap
        )
        s.style_chunk_size = cc.get("style", {}).get("chunk_size", s.style_chunk_size)
        s.style_chunk_overlap = cc.get("style", {}).get(
            "chunk_overlap", s.style_chunk_overlap
        )

        # ── Retrieval ──
        rc = _yaml_config.get("retrieval", {})
        s.fact_top_k = rc.get("fact_top_k", s.fact_top_k)
        s.style_top_k = rc.get("style_top_k", s.style_top_k)

        # ── Guard ──
        gc = _yaml_config.get("guard", {})
        s.guard_max_answer_length = gc.get(
            "max_answer_length", s.guard_max_answer_length
        )
        s.guard_forbidden_words = gc.get("forbidden_words", s.guard_forbidden_words)

        # ── Environment overrides ──
        s.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", s.dashscope_api_key)
        s.app_env = os.getenv("APP_ENV", s.app_env)
        s.log_level = os.getenv("LOG_LEVEL", s.log_level)

        return s


# ── 全局单例 ──────────────────────────────
settings: Settings = Settings.load()
