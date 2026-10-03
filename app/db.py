from pymongo import MongoClient, ASCENDING
import certifi
from app.config import MONGODB_URI, DB_NAME

try:
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
except Exception:
    client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

documents = db["documents"]         # one document per ingested PDF
tables_meta = db["tables_meta"]     # schema catalog: one per logical table
table_rows = db["table_rows"]       # one document per table row
text_chunks = db["text_chunks"]     # narrative chunks + embeddings
baseline_chunks = db["baseline_chunks"]  # flattened page text (baseline RAG)

def ensure_indexes():
    table_rows.create_index([("doc_id", ASCENDING), ("table_id", ASCENDING)])
    tables_meta.create_index(
        [("doc_id", ASCENDING), ("table_id", ASCENDING)],
        unique=True
    )
    text_chunks.create_index([("doc_id", ASCENDING)])
    baseline_chunks.create_index([("doc_id", ASCENDING)])
