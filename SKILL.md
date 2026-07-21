# RAG 企业知识库 — 零环境搭建手册

> 来源：林翩翩 AI 角色对话项目实战经验  
> 向量库：Chroma（本地原型）→ Milvus（企业部署）  
> 目标机器：全新 Windows 电脑，无 Python、无 Docker

---

## 〇、新电脑从零到跑通

### 0.1 装 Python

```powershell
# 1. 下载 Python 3.11（不要 3.12/3.13，部分 AI 包兼容性差）
#    https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
# 2. 安装时勾选 ✅ "Add Python to PATH"
# 3. 验证
python --version   # 应输出 Python 3.11.x
```

### 0.2 装 Docker Desktop（Milvus 需要）

```powershell
# 1. 下载 Docker Desktop
#    https://www.docker.com/products/docker-desktop/
# 2. 安装后启动 Docker Desktop，等待右下角鲸鱼图标变绿
# 3. 验证
docker --version
```

### 0.3 启动 Milvus（一行命令）

```powershell
# 下载 docker-compose.yml 并启动
curl -O https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml
docker-compose up -d

# 验证（等 30 秒后）
docker ps  # 应看到 milvus-standalone 和 etcd 两个容器
```

> 如果公司网络不能下载 GitHub raw，手动创建 `docker-compose.yml`，内容见本文件末尾附录 A。

### 0.4 创建项目 + 安装依赖

```powershell
mkdir C:\rag-project
cd C:\rag-project
python -m venv venv
venv\Scripts\activate

# 核心依赖（一次性安装）
pip install fastapi uvicorn[standard] pymilvus pydantic pydantic-settings
pip install langchain langchain-community langchain-core
pip install dashscope python-dotenv pyyaml
pip install lark docx2txt pypdf     # 文档解析
pip install httpx                    # 异步 HTTP

# 冻结依赖清单
pip freeze > requirements.txt
```

### 0.5 配置 API 密钥

```powershell
# 在项目根目录创建 .env
echo DASHSCOPE_API_KEY=sk-你的阿里云密钥 > .env
echo MILVUS_HOST=localhost >> .env
echo MILVUS_PORT=19530 >> .env
```

> DashScope API Key 获取：https://dashscope.console.aliyun.com/apiKey

### 0.6 跑通验证

```powershell
# 用 Python 测试 Milvus 连接
python -c "
from pymilvus import connections
connections.connect(host='localhost', port='19530')
print('Milvus 连接成功')
"

# 启动 FastAPI
uvicorn app.main:app --reload
# 浏览器打开 http://127.0.0.1:8000/docs 看到 Swagger UI 即成功
```

---

## 一、RAG 管道核心模式

### 1.1 五层保真链

```
数据预处理 → 双通道检索 → Prompt 拼装 → LLM 生成 → 后置校验
```

每一层都是独立防线，任何一层失效都有后续兜底。

### 1.2 数据预处理（最关键的一步）

**原则：按文档性质分类入库，不同类型用不同分块策略。**

```yaml
# 企业知识库场景的分类参考
事实类（chunk_size 偏大 512~1024）:
  - SOP/规章制度 → 需要完整段落，保证上下文完整
  - 产品规格/技术文档 → 表格和数值不能断章取义
  - 合同/法律文本 → 条款完整性优先

风格类（chunk_size 偏小 128~256）:
  - FAQ / 客服话术 → 单条问答，精准匹配
  - 培训案例 → 场景级分块
  - 对话记录 → 单轮对话为单位
```

**每条 chunk 必须带 metadata**：

```python
{
    "doc_type": "sop",        # 文档类型 → 支持按类型过滤检索
    "priority": "high",       # high/medium/low → 检索排序加权
    "source": "财务制度v3.md", # 溯源
    "department": "财务部",    # 业务标签 → 权限控制
    "updated_at": "2026-06",  # 时效性
}
```

### 1.3 双通道检索

**不要把所有文档混在一起检索。按目的分通道：**

```python
# 通道 1：事实/知识检索（保证"说对"）
fact_docs = vector_store.search(
    query, top_k=6,
    filter={"doc_type": {"$in": ["sop", "spec", "contract"]}}
)

# 通道 2：风格/话术检索（保证"说像"）
style_docs = vector_store.search(
    query, top_k=2,
    filter={"doc_type": {"$in": ["faq", "script", "case"]}}
)
```

Prompt 里必须写明：

> 风格检索片段仅用于表达方式和语气参考，不可作为事实依据。  
> 事实以事实检索通道返回的文档为准。

### 1.4 Prompt 分层设计

```
System Prompt（固定）     → 角色定位、边界规则、输出格式
User Context（动态）      → 用户身份、对话历史摘要
RAG Context（检索注入）   → 事实片段 + 风格片段
Current Query（当前）     → 用户最新输入
```

**System Prompt 模板化，运行时填充**：

```
你是 {role_name}。你的职责是 {role_description}。

【知识范围】仅基于提供的检索文档作答，不确定时明确说"不确定"。
【表达规则】{style_rules}
【边界约束】{boundary_rules}
【对话对象】{user_context}
```

### 1.5 后置校验（Guard）

| 检查项 | 方法 | 违规处理 |
|---|---|---|
| 禁止词 | 关键词列表 | 替换为兜底回复 |
| 长度 | 硬截断 | >max_len 截断+省略号 |
| 格式 | 正则 | 确保符合输出模板 |
| 幻觉检测 | 对比检索结果 | 检测到编造则追加免责声明 |
| 合规 | 轻量语义模型（可选） | 标记+人工审核 |

---

## 二、Milvus 企业级 RAG 架构

### 2.1 推荐技术栈

```
FastAPI (异步 Web)
  ├── POST /api/search     向量检索
  ├── POST /api/chat       RAG 对话
  ├── POST /api/ingest     文档入库
  └── POST /api/admin/reindex  索引管理

Milvus (向量数据库)
  ├── Collection: knowledge_base
  │     fields: id, embedding(1024d), text, metadata(JSON)
  │     index: IVF_FLAT / HNSW
  └── Partition: 按部门/项目分 partition

Embedding: text-embedding-v4 (DashScope) / bge-large-zh (本地)
LLM: qwen-plus / qwen-max (阿里云) / 本地部署
```

### 2.2 Milvus 关键配置

```python
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType

# 连接
connections.connect(host="localhost", port="19530")

# Schema 设计
fields = [
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("doc_id", DataType.VARCHAR, max_length=128),       # 源文档 ID
    FieldSchema("chunk_idx", DataType.INT32),                      # 分块序号
    FieldSchema("text", DataType.VARCHAR, max_length=65535),       # 原文
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1024),     # 向量
    FieldSchema("doc_type", DataType.VARCHAR, max_length=32),      # 类型标签
    FieldSchema("metadata", DataType.JSON),                        # 扩展元数据
]

schema = CollectionSchema(fields, description="Enterprise RAG KB")

# 索引
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index("embedding", index_params)
```

### 2.3 FastAPI 路由设计

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Enterprise RAG API")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 6
    doc_types: list[str] | None = None   # 可选过滤
    department: str | None = None         # 可选权限范围

class ChatRequest(BaseModel):
    query: str
    history: list[dict] | None = None     # 对话历史
    doc_types: list[str] | None = None

@app.post("/api/search")
async def search(req: SearchRequest):
    """纯检索，返回文档列表"""
    embedding = embed(req.query)
    filter_expr = build_filter(req.doc_types, req.department)
    results = collection.search(
        [embedding], "embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=req.top_k,
        expr=filter_expr
    )
    return {"results": format_results(results)}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """RAG 对话"""
    docs = search_internal(req.query, req.doc_types)
    prompt = build_prompt(req.query, docs, req.history)
    answer = llm.invoke(prompt)
    return {"answer": answer, "sources": [d["doc_id"] for d in docs]}

@app.post("/api/ingest")
async def ingest(file: UploadFile, doc_type: str, metadata: dict):
    """文档入库"""
    text = extract_text(file)
    chunks = split_text(text, doc_type)
    embeddings = embed_batch([c.text for c in chunks])
    insert_to_milvus(chunks, embeddings, doc_type, metadata)
    return {"chunks": len(chunks)}
```

### 2.4 惰性初始化模式（避免 import 副作用）

```python
# ❌ 旧方式：模块级初始化
# from models import chat_model  # import 时就连接 API

# ✅ 新方式：惰性单例
_llm = None
_embed_model = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatTongyi(model=settings.chat_model_name)
    return _llm

def get_embedder():
    global _embed_model
    if _embed_model is None:
        _embed_model = DashScopeEmbeddings(model=settings.embedding_model_name)
    return _embed_model
```

---

## 三、项目结构建议

```
enterprise-rag/
├── app/
│   ├── main.py              # FastAPI 入口 + 生命周期
│   ├── config.py            # 配置（从 .env / YAML 加载）
│   ├── routes/
│   │   ├── search.py         #   /api/search
│   │   ├── chat.py           #   /api/chat
│   │   └── ingest.py         #   /api/ingest
│   ├── core/
│   │   ├── embeddings.py     #   Embedding 模型管理
│   │   ├── llm.py            #   LLM 管理
│   │   └── milvus.py         #   Milvus 连接管理
│   ├── pipeline/
│   │   ├── splitter.py       #   文档分割器
│   │   ├── retriever.py      #   双通道检索
│   │   ├── prompt.py         #   Prompt 拼装
│   │   └── guard.py          #   后置校验
│   ├── models/
│   │   └── schemas.py        #   Pydantic 请求/响应模型
│   └── middleware/
│       ├── auth.py           #   鉴权
│       └── logging.py        #   日志
├── config/
│   ├── settings.yml
│   └── prompts/
│       ├── system.txt
│       └── rag.txt
├── tests/
├── Dockerfile
├── docker-compose.yml        # Milvus + FastAPI
└── requirements.txt
```

---

## 四、常见坑与解法

| 坑 | 原因 | 解法 |
|---|---|---|
| Milvus 连接超时 | 防火墙/端口未开放 | `docker-compose` 确保 `19530` 端口映射 |
| 检索精度低 | chunk 太大或太小 | 先手工测几个 query，找到最佳 chunk_size |
| Prompt 太长 | 检索 top_k 太大 | 事实 k=6, 风格 k=2 分开控制 |
| LLM 编造事实 | 检索结果不相关 | 加幻觉检测：对比 answer 和 retrieved docs |
| 增量入库重复 | 没有去重机制 | MD5 指纹 or Milvus upsert by doc_id |
| Embedding 维度不匹配 | 模型换了维度变了 | Milvus collection 创建后维度不可变，需重建 |

---

## 五、从 Chroma 迁移到 Milvus 检查清单

1. [ ] `pip install pymilvus`
2. [ ] `docker-compose up -d` 启动 Milvus standalone
3. [ ] 用 `connections.connect()` 测试连接
4. [ ] 设计 Schema（主键、向量维度、标量字段、JSON metadata）
5. [ ] 选索引类型：IVF_FLAT（百万级）/ HNSW（千万级）
6. [ ] 改写 `vector_store.py`：`Chroma.add_documents()` → `Milvus.insert()`
7. [ ] 改写检索：`Chroma.similarity_search()` → `Milvus.search()` + `expr` 过滤
8. [ ] 验证 filter 语法：`doc_type in ["sop","spec"]` 用 Milvus `expr` 写法
9. [ ] 性能基准测试：100 万条 → 检索延迟 < 100ms

---

## 附录 A：docker-compose.yml（Milvus Standalone）

```yaml
version: '3.5'
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ./volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ./volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
```

> 保存到项目根目录，`docker-compose up -d` 即可。

---

## 附录 B：快速搭建命令汇总（复制粘贴用）

```powershell
# === 1. 环境检查 ===
python --version                              # 确保 3.11
docker --version                              # 确保已启动

# === 2. 启动 Milvus ===
docker-compose up -d                          # 等 30 秒
python -c "from pymilvus import connections; connections.connect(host='localhost',port='19530'); print('OK')"

# === 3. 创建项目 ===
mkdir C:\rag-project && cd C:\rag-project
python -m venv venv && venv\Scripts\activate
pip install fastapi uvicorn pymilvus pydantic pydantic-settings langchain langchain-community langchain-core dashscope python-dotenv pyyaml httpx lark docx2txt pypdf

# === 4. 配置密钥 ===
echo DASHSCOPE_API_KEY=sk-你的KEY > .env

# === 5. 复制项目结构（从 SKILL.md 第三章） ===
mkdir app\core,app\routes,app\pipeline,app\models,app\middleware,config\prompts,tests

# === 6. 启动 ===
uvicorn app.main:app --reload
# 打开 http://127.0.0.1:8000/docs
```
