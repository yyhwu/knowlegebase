"""
FastAPI 入口 — 企业 RAG 知识库 API
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.milvus import get_milvus_collection, release_collection
from app.routes import search, chat, ingest, eval, conversations

logger = get_logger(__name__)


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# ── 生命周期管理 ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动 / 关闭钩子"""
    setup_logging()
    logger.info("正在连接 Milvus...")
    try:
        client = get_milvus_collection()
        logger.info("Collection '%s' 已就绪", settings.collection_name)
    except Exception as e:
        logger.warning("Milvus 连接失败: %s", e)

    # 预加载 OCR 模型
    logger.info("正在预加载 PaddleOCR（首次加载约 2-5 秒）...")
    try:
        from app.routes.ingest import _get_ocr
        ocr = _get_ocr()
        if ocr:
            logger.info("PaddleOCR 预加载成功")
        else:
            logger.warning("PaddleOCR 预加载失败，图片型 PDF 将无法识别！")
    except Exception as e:
        logger.warning("PaddleOCR 预加载异常: %s", e)

    # 初始化会话持久层
    logger.info("正在初始化会话存储 (SQLite)...")
    try:
        from app.core.session_sqlite import get_session_store
        store = await get_session_store()
        logger.info("会话存储已就绪 (%s)", store._db_path)
    except Exception as e:
        logger.warning("会话存储初始化失败: %s", e)

    yield

    logger.info("正在释放 Milvus 资源...")
    release_collection()


# ── 创建 App ─────────────────────────────

app = FastAPI(
    title="Enterprise RAG API",
    description="基于 Milvus + DashScope 的企业级 RAG 知识库",
    version="1.0.0",
    lifespan=lifespan,
)

# ── 注册路由 ─────────────────────────────

app.include_router(search.router, prefix="/api", tags=["检索"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(ingest.router, prefix="/api", tags=["入库"])
app.include_router(eval.router, prefix="/api", tags=["评估"])
app.include_router(conversations.router, prefix="/api", tags=["会话"])


# ── 前端页面 ─────────────────────────────

STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
async def index():
    """前端测试页面"""
    return FileResponse(STATIC_DIR / "index.html")


# ── 健康检查 ─────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Enterprise RAG API"}


@app.get("/api/cache-stats")
async def cache_stats():
    """缓存统计"""
    from app.pipeline.cache import get_cache_stats
    return get_cache_stats()


@app.get("/api/cache-stats")
async def cache_stats():
    """缓存统计"""
    from app.pipeline.cache import get_cache_stats
    return get_cache_stats()
