import hashlib
from datetime import datetime, timezone
from app.db import documents, tables_meta, table_rows, text_chunks, baseline_chunks
from app.pdf_parser import parse_pdf, merge_multipage
from app.table_store import save_tables
from app.text_store import save_chunks

def ingest_pdf(path, filename="document.pdf", api_key=None):
    with open(path, "rb") as f:
        doc_id = hashlib.sha1(f.read()).hexdigest()[:16]

    for coll in (tables_meta, table_rows, text_chunks, baseline_chunks):
        coll.delete_many({"doc_id": doc_id})  # idempotent re-ingestion

    text_blocks, raw_tables, page_texts = parse_pdf(path)
    merged = merge_multipage(raw_tables)
    n_rows = save_tables(doc_id, merged)
    n_chunks = save_chunks(doc_id, text_blocks, text_chunks, api_key=api_key)
    n_base = save_chunks(doc_id, page_texts, baseline_chunks, api_key=api_key)

    summary = {
        "doc_id": doc_id,
        "filename": filename,
        "tables": len(merged),
        "table_rows": n_rows,
        "text_chunks": n_chunks,
        "baseline_chunks": n_base,
        "ingested_at": datetime.now(timezone.utc)
    }
    documents.replace_one({"_id": doc_id}, {"_id": doc_id, **summary}, upsert=True)
    summary.pop("ingested_at")
    return summary
