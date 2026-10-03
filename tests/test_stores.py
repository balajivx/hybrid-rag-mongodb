from app.table_store import snake, cast, unique
from app.text_store import chunk_text

def test_snake_and_cast():
    assert snake("Team Name (2022)") == "team_name_2022"
    assert cast("123") == 123
    assert cast("12.34") == 12.34
    assert cast("—") is None
    assert cast("2022-12-18") == "2022-12-18"

def test_unique_column_names():
    cols = unique(["score", "team", "score"])
    assert cols == ["score", "team", "score_2"]

def test_chunking():
    blocks = [{"page": 1, "text": "This is sentence one. This is sentence two. This is sentence three."}]
    chunks = chunk_text(blocks, size=50, overlap=10)
    assert len(chunks) >= 1
    assert chunks[0]["page"] == 1
