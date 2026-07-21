"""查看所有文档"""
import os,sys
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
sys.path.insert(0,r'c:\Users\Administrator\Desktop\knowleggeBase')
from app.core.milvus import get_milvus_collection
from app.config import settings
from collections import Counter

c=get_milvus_collection()
r=c.query(collection_name=settings.collection_name,filter='',output_fields=['doc_id','doc_type','metadata'],limit=500)
print(f'Total chunks: {len(r)}')

docs=Counter()
for d in r:
    did=d['doc_id']
    src=(d.get('metadata') or {}).get('source','?')
    docs[(did,src)]+=1

print(f'Unique docs: {len(docs)}')
for (did,src),cnt in docs.items():
    print(f'  {did[:8]}... chunks={cnt} source={src[:50]}')
