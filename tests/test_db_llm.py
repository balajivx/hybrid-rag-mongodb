from app.db import db, ensure_indexes, documents, tables_meta, table_rows, text_chunks, baseline_chunks

def test_db_collections():
    assert documents.name == "documents"
    assert tables_meta.name == "tables_meta"
    assert table_rows.name == "table_rows"
    assert text_chunks.name == "text_chunks"
    assert baseline_chunks.name == "baseline_chunks"
