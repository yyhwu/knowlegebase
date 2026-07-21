"""快速查询数据库"""
import os,sys
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
sys.path.insert(0,r'c:\Users\Administrator\Desktop\knowleggeBase')
from app.core.milvus import get_milvus_collection
from app.config import settings
c=get_milvus_collection()
r=c.query(collection_name=settings.collection_name,filter='',output_fields=['doc_id','metadata'],limit=10)
print(f'Rows: {len(r)}')
for d in r:
    src=(d.get('metadata') or {}).get('source','?')
    print(f'  {d["doc_id"][:8]}... source={src}')
