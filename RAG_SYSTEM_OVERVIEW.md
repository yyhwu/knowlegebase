# 企业 RAG 知识库 — 技术架构与优化总结

> **项目**: Enterprise RAG API v1.0.0  
> **框架**: FastAPI + Milvus + DashScope (阿里云)  
> **更新日期**: 2026-07-17

---

## 目录

1. [技术栈](#1-技术栈)
2. [架构设计](#2-架构设计)
3. [环境依赖版本](#3-环境依赖版本)
4. [RAG 优化策略汇总](#4-rag-优化策略汇总)
   - [4.1 检索层优化](#41-检索层优化)
   - [4.2 文档处理层优化](#42-文档处理层优化)
   - [4.3 生成层优化](#43-生成层优化)
   - [4.4 性能层优化](#44-性能层优化)
   - [4.5 评估层优化](#45-评估层优化)
5. [已知问题与约束](#5-已知问题与约束)

---

## 1. 技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| **Web 框架** | FastAPI + uvicorn | 异步 API 服务，lifespan 生命周期管理 |
| **向量数据库** | Milvus (pymilvus 3.x) | HNSW 密集向量 + SPARSE_INVERTED_INDEX 稀疏向量双索引 |
| **Embedding 模型** | DashScope `text-embedding-v4` | 1024 维向量，通过 `langchain_community` 调用 |
| **LLM** | 通义千问 (`qwen-turbo`/`qwen-plus`) | 通过 `langchain ChatTongyi` 调用，temperature=0.1 |
| **Reranker** | DashScope `gte-rerank` | Cross-encoder 精排，DashScope API 直接调用 |
| **OCR 引擎** | PaddleOCR 2.9.1 + pypdfium2 | 扫描件/图片型 PDF 文字提取 |
| **PDF 解析** | pypdf → pdfplumber → PaddleOCR | 三级降级链路，自适应电子档/扫描件 |
| **中文分词** | jieba + 同义词词典 | Query 扩展后喂入 Milvus 内置分词器做 BM25 检索 |
| **文本分割** | langchain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | Markdown 结构感知 + 递归段落切分 |
| **日志系统** | Python `logging` + `TimedRotatingFileHandler` | stdout + 文件双输出，按天轮转，保留 30 天 |
| **会话存储** | SQLite (`aiosqlite`) | WAL 模式，预留 PostgreSQL / Redis 抽象接口 |
| **配置管理** | Pydantic Settings + YAML + `.env` | `${ENV:default}` 占位符解析，环境变量覆盖 |
| **数据校验** | Pydantic v2 | 请求/响应模型 |
| **前端** | 原生 HTML/CSS/JS | 单页应用（上传 + 对话 + 评估面板） |

---

## 2. 架构设计

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────┐
│                     前端 (static/index.html)                  │
│               上传文件 · RAG 对话 · 评估面板                   │
└──────────────────────────┬───────────────────────────────────┘
                           │ HTTP REST
┌──────────────────────────┴───────────────────────────────────┐
│                    FastAPI (app/main.py)                      │
│  ┌──────────┬──────────┬──────────┬──────────┬───────────┐  │
│  │ /ingest  │ /search  │  /chat   │  /eval   │ /conversations │
│  │ 文档入库  │ 纯检索   │ RAG 对话 │ 质量评估  │ 会话 CRUD  │  │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴─────┬─────┘  │
│       │          │          │          │           │         │
│  ┌────┴──────────┴──────────┴──────────┴───────────┴────┐    │
│  │                 pipeline/ — RAG 处理管道                │    │
│  │                                                        │    │
│  │  splitter ── query_processor ── retriever ── reranker │    │
│  │  prompt ──── guard ────────────── cache ──── eval_*    │    │
│  │  query_expansion                                       │    │
│  └──────────────────────┬───────────────────────────────┘    │
│                         │                                     │
│  ┌──────────────────────┴───────────────────────────────┐    │
│  │                   core/ — 基础服务层                    │    │
│  │                                                        │    │
│  │  embeddings (DashScope)  │  llm (通义千问)             │    │
│  │  milvus (Schema+Index)   │  session (SQLite)           │    │
│  │  logging (stdout+文件)   │                              │    │
│  └──────────────────────┬───────────────────────────────┘    │
└─────────────────────────┼────────────────────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌──────────┐    ┌──────────────┐    ┌─────────────────┐
│  Milvus  │    │ DashScope API │    │  SQLite (data/) │
│ 向量数据库│    │ Embedding/LLM │    │  会话持久化      │
│          │    │ /Reranker     │    │                 │
└──────────┘    └──────────────┘    └─────────────────┘
```

### 2.2 目录结构

```
knowleggeBase/
├── app/
│   ├── main.py                  # FastAPI 入口 + lifespan 生命周期
│   ├── config.py                # 配置中心 (YAML + .env → Settings 单例)
│   │
│   ├── core/                    # 基础服务层
│   │   ├── embeddings.py        #   Embedding 抽象 (DashScope / local 预留)
│   │   ├── llm.py               #   LLM 抽象 (通义千问 / vLLM 预留)
│   │   ├── milvus.py            #   Milvus Schema、索引、连接管理
│   │   ├── logging.py           #   集中日志配置 (stdout + 文件轮转)
│   │   ├── session.py           #   会话持久层抽象接口 (ABC)
│   │   └── session_sqlite.py    #   SQLite 实现 (aiosqlite, WAL 模式)
│   │
│   ├── models/
│   │   └── schemas.py           #   Pydantic 请求/响应模型
│   │
│   ├── pipeline/                # RAG 业务管道
│   │   ├── splitter.py          #   文档分割 (Markdown 结构感知 + 父子 Chunk)
│   │   ├── query_processor.py   #   Query 预处理 (HyDE / 拆解 / 上下文补全)
│   │   ├── query_expansion.py   #   零 LLM 关键词扩展 (jieba + 同义词)
│   │   ├── retriever.py         #   混合检索 (Dense + BM25 稀疏向量 → RRF → Reranker)
│   │   ├── reranker.py          #   Cross-encoder 精排 (DashScope gte-rerank)
│   │   ├── prompt.py            #   RAG Prompt 拼装 (System + 历史 + 上下文)
│   │   ├── guard.py             #   后置校验 (截断 / 兜底 / 幻觉标记)
│   │   ├── cache.py             #   LRU 缓存 (Embedding + 检索结果)
│   │   ├── eval_dataset.py      #   评估数据集自动生成
│   │   ├── eval_retrieval.py    #   检索评估 (MRR / NDCG@k / Precision@k)
│   │   └── eval_generation.py   #   生成评估 (Faithfulness / Answer Relevance)
│   │
│   └── routes/                  # API 路由层
│       ├── ingest.py            #   文档入库 (TXT / MD / PDF / DOCX / XLSX)
│       ├── search.py            #   纯检索接口
│       ├── chat.py              #   RAG 对话 (检索 → 生成 → Guard → 持久化)
│       ├── eval.py              #   评估接口 (数据集 / 检索 / 生成 / 全量)
│       └── conversations.py     #   会话 CRUD 接口
│
├── config/
│   ├── settings.yml             #   全局 YAML 配置
│   └── prompts/
│       └── system.txt           #   System Prompt 模板
│
├── data/
│   └── sessions.db              #   SQLite 会话数据库 (运行时生成)
│
├── logs/
│   └── app.log                  #   应用日志 (按天轮转，保留 30 天)
│
├── static/
│   └── index.html               #   前端单页应用
│
├── tests/
│   ├── eval_data/
│   │   └── eval_dataset.json   #   评估数据集 (query + gt_doc + reference_answer)
│   └── ...                      #   各类测试/调试脚本
│
├── docker-compose.yml           #   Milvus Standalone (etcd + MinIO + Milvus)
├── requirements.txt             #   Python 依赖清单
└── .env                         #   环境变量 (API Key 等)
```

### 2.3 数据流 — RAG 对话完整链路

```
用户输入 (query)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 1. Query 预处理 (query_processor.py)                 │
│    检测: 模糊查询? 复杂查询? 多轮上下文?              │
│    按需 LLM 增强: HyDE 假答案 / 拆解子问题 / 上下文补全│
│    输出: dense_queries[] + bm25_query (扩展文本)      │
└──────────────────────────┬──────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    ▼                      ▼                      ▼
┌───────────┐    ┌──────────────┐    ┌──────────────────┐
│ 2a. 密集   │    │ 2b. BM25     │    │ 2c. 去重 + 合并  │
│ 向量检索   │    │ 稀疏向量检索  │    │ (doc_id+chunk)  │
│ COSINE     │    │ Milvus 分词  │    │                  │
│ ef=64      │    │              │    │                  │
└─────┬─────┘    └──────┬───────┘    └────────┬─────────┘
      │                 │                     │
      └────────┬────────┘                     │
               ▼                              │
┌──────────────────────────────┐              │
│ 3. RRF 融合 (k=60)           │              │
│    Reciprocal Rank Fusion    │◄─────────────┘
│    密集排名 + BM25 排名       │
│    → 融合 Top-K              │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 4. Reranker 精排              │
│    DashScope gte-rerank      │
│    Cross-encoder 二次打分     │
│    → Top-N (默认5)            │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 5. 父文档上下文扩展            │
│    同 parent 的小 chunk 合并  │
│    还原章节级语义片段          │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 6. Prompt 拼装                │
│    System Prompt              │
│    + 对话历史 (最近6轮)        │
│    + 检索上下文 (带标题+相关度) │
│    + 当前用户问题              │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 7. LLM 生成                   │
│    通义千问 qwen-turbo        │
│    temperature=0.1            │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 8. Guard 后置校验              │
│    长度截断 / 空回复兜底       │
│    / 幻觉风险标记              │
└──────────────┬───────────────┘
               ▼
         最终回答 + 引用来源
```

### 2.4 Milvus Collection Schema

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64 | 主键，auto_id |
| `doc_id` | VARCHAR(128) | 文档唯一标识 |
| `chunk_idx` | INT32 | 分片序号 |
| `text` | VARCHAR(65535) | 分片文本，enable_analyzer |
| `embedding` | FLOAT_VECTOR(1024) | DashScope 向量 |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | BM25 自动生成 (Function) |
| `doc_type` | VARCHAR(32) | 文档类型 (sop/spec/faq/...) |
| `parent_idx` | INT32 | 父 chunk 索引 |
| `heading_title` | VARCHAR(256) | 所属标题文本 |
| `heading_level` | INT32 | 标题层级 (0-4) |
| `metadata` | JSON | 扩展元数据 (priority/source/department/tags) |

**索引**:
- `embedding`: HNSW, COSINE, M=16, efConstruction=200
- `sparse_vector`: SPARSE_INVERTED_INDEX, IP

---

## 3. 环境依赖版本

### 3.1 核心依赖

| 包名 | 版本 | 用途 | 约束说明 |
|------|------|------|----------|
| `pymilvus` | 3.0.0+ | MilvusClient 模式连接 | 需要 protobuf >=5 |
| `PaddlePaddle` | 2.6.2 | OCR 底层计算框架 | ⚠️ **不可升 3.x** (oneDNN 不兼容 PP-OCRv4) |
| `PaddleOCR` | 2.9.1 | 扫描件/图片型 PDF 识别 | 依赖 PaddlePaddle 2.x |
| `pypdfium2` | latest | PDF 页面渲染 → PaddleOCR 输入 | - |
| `numpy` | <2 (1.26.4) | 数值计算 | ⚠️ PaddleOCR 的 imgaug 兼容约束 |
| `protobuf` | >=5 | pymilvus 3.x 序列化 | 与 PaddlePaddle 2.x 冲突 |
| `langchain` | latest | LLM 编排框架 | - |
| `langchain-community` | latest | DashScope Embeddings / ChatTongyi | - |
| `dashscope` | latest | Reranker API 直接调用 | - |
| `pypdf` | latest | PDF 第一优先级文字提取 | - |
| `pdfplumber` | latest | PDF 第二优先级 (保留布局+表格) | - |
| `docx2txt` | latest | DOCX 文档解析 | - |
| `pandas` + `openpyxl` | latest | Excel 文件解析 | - |
| `jieba` | latest | 中文分词 (零 LLM 关键词扩展) | - |
| `aiosqlite` | latest | 异步 SQLite 会话存储 | WAL 模式 |
| `fastapi` | latest | Web 框架 | - |
| `uvicorn` | latest | ASGI 服务器 | - |
| `pydantic` | v2 | 数据校验 | - |
| `pydantic-settings` | latest | 配置管理 | - |
| `pyyaml` | latest | YAML 配置解析 | - |

### 3.2 环境变量要求

```bash
# 必须设置 (解决 PaddlePaddle + pymilvus protobuf 冲突)
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# API 密钥
DASHSCOPE_API_KEY=sk-xxx

# Milvus 连接
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 3.3 基础设施 (Docker)

| 服务 | 镜像 | 端口 |
|------|------|------|
| Milvus Standalone | milvusdb/milvus:v2.4.23 | 19530 |
| etcd | quay.io/coreos/etcd:v3.5.5 | 2379 |
| MinIO | minio/minio:RELEASE.2023-03-20 | 9000 |

---

## 4. RAG 优化策略汇总

### 4.1 检索层优化

| # | 优化策略 | 文件 | 核心逻辑 |
|---|----------|------|----------|
| 1 | **Dense + BM25 稀疏向量混合检索** | `retriever.py` | 语义匹配 + Milvus 内置 BM25 分词评分双路召回 |
| 2 | **RRF 融合** | `retriever.py:109` | Reciprocal Rank Fusion (k=60)，融合两路排名 |
| 3 | **Reranker 精排** | `reranker.py` | DashScope `gte-rerank` Cross-encoder 二次打分 |
| 4 | **HyDE 假答案检索** | `query_processor.py:48` | 模糊查询时 LLM 生成假设答案，用答案 Embedding 检索 |
| 5 | **Query 拆解** | `query_processor.py:66` | 复杂查询拆为独立子问题，多路检索去重合并 |
| 6 | **上下文补全** | `query_processor.py:80` | 多轮对话中指代词替换为具体实体 |
| 7 | **零 LLM 关键词扩展** | `query_expansion.py` | jieba 分词 + 同义词 → 拼成 BM25 查询文本 → Milvus 分词器处理 |
| 8 | **按需 LLM 调用** | `query_processor.py:100` | 检测触发条件后才调 LLM（最多 2 次），简单查询零开销 |
| 9 | **多 Query 去重合并** | `retriever.py:156` | 按 `doc_id+chunk_idx` 去重，每条 query 独立检索 |
| 10 | **父文档上下文扩展** | `retriever.py:205` | 同 parent 的相邻 chunk 合并还原章节语义 |
| 11 | **元数据过滤** | `retriever.py:25` | 按 `doc_type` / `department` 精确过滤 |

### 4.2 文档处理层优化

| # | 优化策略 | 文件 | 核心逻辑 |
|---|----------|------|----------|
| 1 | **Markdown 结构感知分块** | `splitter.py` | 按 h1~h4 标题层级切为 sections |
| 2 | **小 Chunk + 父文档模式** | `splitter.py:20` | Chunk (256 tokens) 入库检索，ParentChunk 提供完整章节上下文 |
| 3 | **表格行保护** | `splitter.py:126` | 检测 `|...|` 格式的表格行，保持完整不切割 |
| 4 | **中文段落分割** | `splitter.py:59` | 分隔符优先级: `\n\n` → `\n` → `。` → `；` → `，` |
| 5 | **PDF 三级降级提取** | `ingest.py:70` | pypdf (电子档) → pdfplumber (布局+表格) → PaddleOCR (扫描件) |
| 6 | **Excel 多 Sheet 解析** | `ingest.py:185` | pandas 读取全部 Sheet，每个 Sheet 截取前 500 行 |
| 7 | **文档去重** | `ingest.py:246` | 按 `metadata.source` (文件名) 检查重复上传 |
| 8 | **文件数量限制** | `ingest.py:253` | 最多 50 个文档 |
| 9 | **多格式支持** | `ingest.py:24` | TXT / MD / PDF / DOCX / CSV / XLSX / JSON / YAML |
| 10 | **BM25 稀疏向量自动生成** | `milvus.py:59` | Milvus Function 入库时自动从 `text` 分词计算 BM25 稀疏向量 |

### 4.3 生成层优化

| # | 优化策略 | 文件 | 核心逻辑 |
|---|----------|------|----------|
| 1 | **System Prompt 约束** | `system.txt` | 角色定位 + 知识边界 + 引用要求 + 防幻觉 + 边界约束 |
| 2 | **对话历史注入** | `prompt.py:42` | 最近 6 轮对话拼入 Prompt |
| 3 | **检索上下文带元数据** | `prompt.py:56` | 每个 doc 展示文档类型、标题路径、相关度分数、片段数量 |
| 4 | **低温度生成** | `config.py:55` | temperature=0.1，保证输出稳定性和一致性 |
| 5 | **后置 Guard 校验** | `guard.py` | 长度截断 (8000 字) + 空回复兜底 + 幻觉风险短语检测 |
| 6 | **引用溯源** | `chat.py:80` | 返回引用的 `doc_id` 列表，前端可溯源 |

### 4.4 性能层优化

| # | 优化策略 | 文件 | 核心逻辑 |
|---|----------|------|----------|
| 1 | **Embedding 缓存 (LRU)** | `cache.py:45` | 2048 条, TTL 10min, MD5 key |
| 2 | **检索结果缓存 (LRU)** | `cache.py:46` | 256 条, TTL 2min |
| 3 | **jieba 分词缓存** | `query_expansion.py:30` | `@lru_cache(maxsize=128)` |
| 4 | **OCR 惰性单例** | `ingest.py:28` | 首次调用加载模型，后续复用 |
| 5 | **启动预加载** | `main.py:31` | lifespan 中预加载 OCR 模型 (2-5s) |
| 6 | **惰性单例模式** | 全局 | Embedder / LLM / MilvusClient / SessionStore 全部惰性加载 |
| 7 | **SQLite WAL 模式** | `session_sqlite.py:28` | Write-Ahead Logging 提升并发读写 |
| 8 | **Milvus HNSW 索引** | `milvus.py:71` | M=16, efConstruction=200, COSINE, 高召回精度 |
| 9 | **Reranker 降级容错** | `retriever.py:196` | Reranker 失败时保留 RRF 排序结果 |
| 10 | **结构化日志** | `core/logging.py` | stdout + 文件双输出，按天轮转保留 30 天，按级别过滤 |

### 4.5 评估层优化

| # | 评估指标 | 文件 | 说明 |
|---|----------|------|------|
| 1 | **MRR** | `eval_retrieval.py:41` | 第一个相关文档的排名倒数平均值 |
| 2 | **NDCG@k** | `eval_retrieval.py:51` | 考虑相关度等级和位置的归一化折损累积增益 |
| 3 | **Precision@k** | `eval_retrieval.py:47` | 前 k 个结果中相关文档占比 |
| 4 | **Faithfulness** | `eval_generation.py:10` | LLM-as-Judge: 答案陈述是否有检索内容依据 |
| 5 | **Answer Relevance** | `eval_generation.py:43` | LLM-as-Judge: 答案是否切题、完整 |
| 6 | **自动数据集生成** | `eval_dataset.py:17` | LLM 从入库文档自动生成 (query, gt_doc, reference_answer) 三元组 |

### 4.6 优化技术全景

```
                          ┌──────────────────────┐
                          │   Query 预处理        │
                          │   HyDE / 拆解 / 补全  │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
        ┌───────────┐        ┌──────────────┐       ┌──────────────┐
        │ Dense 检索 │        │  BM25 检索    │       │ BM25 查询文本 │
        │ COSINE     │        │  稀疏向量 IP  │       │ jieba+同义词  │
        └─────┬─────┘        └──────┬───────┘       └──────────────┘
              │                     │
              └────────┬────────────┘
                       ▼
              ┌────────────────┐
              │  RRF 融合 (k=60)│
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │  Reranker 精排  │  ← gte-rerank Cross-encoder
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ 父文档上下文扩展 │  ← 小 chunk → 章节级语义还原
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │  Prompt 拼装    │  ← System + 历史 + 上下文(带标题/相关度)
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │  LLM 生成       │  ← qwen-turbo, t=0.1
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │  Guard 后置校验 │  ← 截断/兜底/幻觉标记
              └────────────────┘
```

---

## 5. 已知问题与约束

| 问题 | 影响 | 优先级 | 解决方案 |
|------|------|--------|----------|
| **PaddlePaddle 2.6.2 与 pymilvus 3.x protobuf 冲突** | 不设环境变量时 OCR 初始化报错 | 🔴 高 | 启动时设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` |
| **numpy 降级到 1.26.4** | 与 PaddleOCR 的 imgaug 兼容 | 🟡 中 | 保持 numpy <2，不影响 pymilvus |
| **OCR 首次初始化 ~2-5s** | 第一个 PDF 上传稍慢 | 🟢 低 | 启动预加载，后续复用单例 |
| **缓存冷启动** | 系统刚启动时缓存为空，Cache Miss 100% | 🟢 低 | 正常现象，多次查询后自动命中 |
| **不支持加密 PDF** | 加密/受保护 PDF 解析失败 | 🟡 中 | 上传前需解密 |
| **Excel 限 500 行** | 超大 Excel 被截断 | 🟢 低 | 防止 OOM，可通过配置调整 |
| **最多 50 个文档** | 超过限制需先清理 | 🟢 低 | 可通过 `MAX_FILE_COUNT` 配置 |

---

## 附录: API 路由一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/api/cache-stats` | 缓存命中统计 |
| `POST` | `/api/ingest` | 文档入库 |
| `POST` | `/api/search` | 纯检索 (无 LLM) |
| `POST` | `/api/chat` | RAG 对话 |
| `POST` | `/api/eval/generate-dataset` | 生成评估数据集 |
| `GET` | `/api/eval/retrieval` | 检索质量评估 |
| `GET` | `/api/eval/generation` | 生成质量评估 |
| `POST` | `/api/eval/full` | 全链路评估 |
| `POST` | `/api/conversations` | 新建会话 |
| `GET` | `/api/conversations` | 列出会话 |
| `GET` | `/api/conversations/{id}` | 获取会话详情 |
| `POST` | `/api/conversations/{id}/feedback` | 消息反馈 |
| `DELETE` | `/api/conversations/{id}` | 删除会话 |
