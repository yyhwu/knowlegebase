"""
缓存层 — Embedding 缓存 + 检索结果 LRU 缓存
减少重复 embedding 计算和 Milvus 查询
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any


class LRUCache:
    """简单 LRU 缓存"""

    def __init__(self, max_size: int = 512, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        timestamp, value = self._cache[key]
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (time.time(), value)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def stats(self) -> dict:
        return {"size": len(self._cache), "max": self.max_size, "ttl": self.ttl}


# ── 全局缓存实例 ──────────────────────────

embedding_cache = LRUCache(max_size=2048, ttl_seconds=600)   # 10min
search_cache = LRUCache(max_size=256, ttl_seconds=120)        # 2min


def cache_key(texts: str | list[str]) -> str:
    """生成缓存 key"""
    if isinstance(texts, list):
        texts = "|".join(texts)
    return hashlib.md5(texts.encode()).hexdigest()


def get_cached_embedding(text: str) -> list[float] | None:
    return embedding_cache.get(cache_key(text))


def set_cached_embedding(text: str, vector: list[float]) -> None:
    embedding_cache.set(cache_key(text), vector)


def get_cached_search(query: str, doc_types: str, department: str) -> list[dict] | None:
    key = cache_key(f"{query}|{doc_types}|{department}")
    return search_cache.get(key)


def set_cached_search(query: str, doc_types: str, department: str, results: list[dict]) -> None:
    key = cache_key(f"{query}|{doc_types}|{department}")
    search_cache.set(key, results)


def get_cache_stats() -> dict:
    return {
        "embedding": embedding_cache.stats(),
        "search": search_cache.stats(),
    }
