"""
集中日志配置 — stdout + 按天轮转文件，统一替换项目中的 print()
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from app.config import settings

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

# 日志格式：时间 | 级别 | 模块 | 消息
CONSOLE_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
FILE_FORMAT = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_initialized = False


def setup_logging() -> None:
    """配置全局日志（应用启动时调用一次）"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    level = _resolve_level(settings.log_level)

    root = logging.getLogger()
    root.setLevel(level)

    # ── 1. stdout handler ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(CONSOLE_FORMAT)
    root.addHandler(console)

    # ── 2. 文件 handler（按天轮转，保留 30 天）──
    LOG_DIR.mkdir(exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        filename=LOG_DIR / "app.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(FILE_FORMAT)
    root.addHandler(file_handler)

    # ── 3. 按模块级别静默第三方库 ──
    for noisy in ("uvicorn.access", "pymilvus", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 自身
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统已初始化 (level={settings.log_level}, dir={LOG_DIR})")


def _resolve_level(level_str: str) -> int:
    return getattr(logging, level_str.upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger（触发惰性初始化）"""
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)
