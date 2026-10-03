from app.pdf_parser import merge_multipage, _continues

def test_merge_multipage_continuation():
    raw_tables = [
        {
            "page": 1,
            "bbox": (50, 200, 500, 750),
            "page_height": 800,
            "rows": [["Col1", "Col2"], ["Val1", "Val2"]],
            "caption": "Table 1: Example"
        },
        {
            "page": 2,
            "bbox": (50, 50, 500, 700),
            "page_height": 800,
            "rows": [["Col1", "Col2"], ["Val3", "Val4"]],
            "caption": None
        }
    ]
    merged = merge_multipage(raw_tables)
    assert len(merged) == 1
    assert merged[0]["caption"] == "Table 1: Example"
    assert merged[0]["pages"] == [1, 2]
    assert len(merged[0]["rows"]) == 2  # ["Val1", "Val2"], ["Val3", "Val4"]
