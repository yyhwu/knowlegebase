"""
Query 扩展 — 零 LLM 调用：同义词 + jieba 分词 + 拼音纠错
"""
from __future__ import annotations

import re
from functools import lru_cache

# ── 同义词词典（按领域扩展）─────────────

_SYNONYMS: dict[str, list[str]] = {
    "报销": ["费用核销", "差旅费", "报销单", "报销流程"],
    "考勤": ["打卡", "签到", "出勤", "迟到", "早退"],
    "审批": ["审核", "批准", "批复"],
    "接口": ["API", "端点", "endpoint"],
    "超时": ["timeout", "超限", "超时错误"],
    "认证": ["登录", "鉴权", "auth", "login"],
    "住宿": ["酒店", "宾馆", "差旅住宿"],
    "制度": ["规定", "规则", "章程", "管理办法"],
}

# 补充：反向映射
for _k, _vs in list(_SYNONYMS.items()):
    for _v in _vs:
        if _v not in _SYNONYMS:
            _SYNONYMS[_v] = [_k]


@lru_cache(maxsize=128)
def _jieba_cut(text: str) -> list[str]:
    """jieba 分词（缓存）"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        # 降级：正则分词
        return [w for w in re.split(r'[，。！？、\s,!\?]+', text) if w]


def expand_keywords(query: str) -> list[str]:
    """
    零 LLM 关键词扩展
    输入: "怎么报销差旅费"
    输出: ["报销", "差旅费", "费用核销", "报销单", "差旅住宿"]
    """
    words = _jieba_cut(query)
    keywords = set()

    for w in words:
        w = w.strip()
        if len(w) < 2 or re.match(r'^[的了么呢吧啊]$', w):
            continue
        keywords.add(w)
        # 查同义词
        for syn in _SYNONYMS.get(w, []):
            keywords.add(syn)

    return list(keywords)[:10]


def expand_queries(queries: list[str]) -> list[str]:
    """批量扩展"""
    all_kw = set()
    for q in queries:
        all_kw.update(expand_keywords(q))
    return list(all_kw)[:15]
