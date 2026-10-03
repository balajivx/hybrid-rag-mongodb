import pytest
from app.retrieval import run_table_pipeline

def test_blocked_stages():
    with pytest.raises(ValueError):
        run_table_pipeline("doc1", "table_1", [{"$out": "other_coll"}])
    with pytest.raises(ValueError):
        run_table_pipeline("doc1", "table_1", [{"$where": "this.a == 1"}])
    with pytest.raises(ValueError):
        run_table_pipeline("doc1", "table_1", [{"$lookup": {"from": "other"}}])
    with pytest.raises(ValueError):
        run_table_pipeline("doc1", "table_1", [{"$match": {"$where": "this.a"}}])
