"""测试 Query 预处理"""
from app.pipeline.query_processor import process_query, get_hybrid_queries

print("=" * 60)
print("Query 预处理测试")
print("=" * 60)

# 测试1：简单查询（零 LLM 调用）
print("\n1. 简单查询：")
result = process_query("报销流程是什么")
print(f"   输入: 报销流程是什么")
print(f"   输出: {result}")
print(f"   LLM调用: 0 次")

# 测试2：模糊查询（触发 HyDE，1次 LLM）
print("\n2. 模糊查询：")
result = process_query("那个东西怎么搞")
print(f"   输入: 那个东西怎么搞")
print(f"   触发: 模糊检测 is_fuzzy=True")
print(f"   输出: {[r[:30]+'...' for r in result]}")

# 测试3：复杂查询（触发拆解，1次 LLM）
print("\n3. 复杂查询：")
result = process_query("考勤制度和报销流程有什么区别，分别怎么操作")
print(f"   输入: 考勤制度和报销流程有什么区别，分别怎么操作")
print(f"   触发: 复杂检测 is_complex=True")
print(f"   输出子问题数: {len(result)}")

# 测试4：多轮对话上下文补全
print("\n4. 多轮上下文补全：")
history = [
    {"role": "user", "content": "ERP-MOD-7 接口怎么调用？"},
    {"role": "assistant", "content": "使用 GET /api/v2/data/query?endpoint=ERP-MOD-7"},
]
result = process_query("它的超时错误码是什么", history)
print(f"   输入: 它的超时错误码是什么")
print(f"   历史: 上一轮在问 ERP-MOD-7")
print(f"   触发: needs_context=True")
print(f"   输出: {result}")

# 测试5：混合查询配置
print("\n5. 混合查询配置：")
qc = get_hybrid_queries("怎么登录系统")
print(f"   dense_queries: {qc['dense_queries']}")
print(f"   bm25_keywords: {qc['bm25_keywords']}")
print(f"   rewritten: {qc['rewritten']}")
