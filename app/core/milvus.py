"""
Milvus 连接与 Collection 管理（惰性单例，pymilvus 3.x）
"""
from pymilvus import MilvusClient, DataType, Function, FunctionType

from app.config import settings


# ── 惰性单例 ──────────────────────────────
_client: MilvusClient | None = None
_collection_ready: bool = False


def _get_client() -> MilvusClient:
    """获取 MilvusClient 实例"""
    global _client
    if _client is None:
        _client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    return _client


def get_milvus_collection() -> MilvusClient:
    """
    获取或初始化 Milvus Collection。
    首次调用时自动创建 Schema + HNSW 索引（如不存在）。
    返回 MilvusClient 实例，用于后续 insert / search 操作。
    """
    global _collection_ready
    client = _get_client()
    collection_name = settings.collection_name

    if _collection_ready:
        return client

    # 如果 collection 已存在，标记就绪
    if client.has_collection(collection_name):
        _collection_ready = True
        return client

    # 使用 Schema 方式创建（确保自定义字段生效）
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
    schema.add_field("chunk_idx", DataType.INT32)
    schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dimension)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=32)
    schema.add_field("parent_idx", DataType.INT32)
    schema.add_field("heading_title", DataType.VARCHAR, max_length=256)
    schema.add_field("heading_level", DataType.INT32)
    schema.add_field("metadata", DataType.JSON)
    # ── BM25 稀疏向量（自动从 text 生成）──
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR, nullable=True)

    # BM25 Function
    bm25_fn = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
    )
    schema.add_function(bm25_fn)

    client.create_collection(collection_name=collection_name, schema=schema)

    # ── 索引：密集 HNSW + 稀疏 BM25 ──
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
    )
    client.create_index(
        collection_name=collection_name,
        index_params=index_params,
    )

    # 加载到内存（必须，否则无法检索）
    client.load_collection(collection_name)

    _collection_ready = True
    return client


def release_collection() -> None:
    """释放资源（优雅关闭）"""
    global _client, _collection_ready
    if _client is not None:
        _client.close()
        _client = None
    _collection_ready = False
