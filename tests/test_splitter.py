"""测试新分块策略"""
from app.pipeline.splitter import split_structured

md_text = """# 员工手册

## 第一章 考勤制度

员工应按时上下班，不得无故迟到早退。
每日工作时间：9:00-18:00，午休 12:00-13:00。
请假需提前一天提交申请，经直属领导批准。

## 第二章 报销流程

差旅报销需在出差结束后 3 个工作日内提交。
报销单据包括：交通票据、住宿发票、餐饮票据。
单次报销金额超过 5000 元须部门总监审批。

### 2.1 交通报销

高铁二等座及以下实报实销，飞机经济舱需提前申请。

### 2.2 住宿报销

一线城市住宿标准 500 元/晚，其他城市 350 元/晚。"""

chunks, parents = split_structured(md_text, doc_type="sop")
print(f"小chunk数: {len(chunks)}")
print(f"父chunk数: {len(parents)}")
print()

for p in parents:
    print(f"父chunk[{p.chunk_idx}] h{p.heading_level} \"{p.heading_title}\": {len(p.text)}字")

print()
for c in chunks:
    print(f"子chunk[{c.chunk_idx}] -> parent={c.parent_idx} h{c.heading_level} \"{c.heading_title}\": {c.text[:60]}...")
