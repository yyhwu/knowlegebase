"""清空 Milvus collection 并重建（使用项目本身的 schema）"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sys
sys.path.insert(0, r'c:\Users\Administrator\Desktop\knowleggeBase')

from pymilvus import MilvusClient
from app.config import settings

client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

# 1. 删除旧 collection
if client.has_collection(settings.collection_name):
    stats = client.get_collection_stats(settings.collection_name)
    print(f"清空前: {stats.get('row_count', 0)} 行")
    client.drop_collection(settings.collection_name)
    print(f"已删除: {settings.collection_name}")
else:
    print("Collection 不存在，无需清空")

# 2. 重启 uvicorn 会自动重建
client.close()
print("完成！重启 uvicorn 后，首次访问会自动创建新 collection。")
