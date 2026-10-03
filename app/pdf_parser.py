import re
import pdfplumber
from app.config import MIN_TABLE_ROWS

MARGIN = 40  # points; running header/footer band
CAPTION = re.compile(r"^Table\s+\d+\s*[:.\-–]\s*.+", re.I)

def _clean(cell):
    return re.sub(r"\s+", " ", cell or "").strip()

def _caption_above(page, bbox):
    """Look for a 'Table N: ...' line in the 60pt band above the table."""
    top = bbox[1]
    band = page.crop((0, max(0, top - 60), page.width, top))
    for line in reversed((band.extract_text() or "").splitlines()):
        if CAPTION.match(line.strip()):
            return line.strip()
    return None

def parse_pdf(path):
    """Return (text_blocks, raw_tables, page_texts)."""
    text_blocks, raw_tables, page_texts = [], [], []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            page_texts.append({"page": pno, "text": page.extract_text() or ""})
            data_bboxes = []
            for t in page.find_tables():
                rows = [[_clean(c) for c in r] for r in t.extract()]
                rows = [r for r in rows if any(r)]
                if len(rows) < MIN_TABLE_ROWS or len(rows[0]) < 2:
                    continue  # small grids stay in the text path
                data_bboxes.append(t.bbox)
                raw_tables.append({
                    "page": pno,
                    "bbox": t.bbox,
                    "page_height": page.height,
                    "rows": rows,
                    "caption": _caption_above(page, t.bbox),
                })

            def keep(obj):  # drop table cells, running header and footer
                cx = (obj["x0"] + obj["x1"]) / 2
                cy = (obj["top"] + obj["bottom"]) / 2
                in_margin = cy < MARGIN or cy > page.height - MARGIN
                in_table = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in data_bboxes)
                return not (in_margin or in_table)

            text = page.filter(keep).extract_text() or ""
            if text.strip():
                text_blocks.append({"page": pno, "text": text})

    return text_blocks, raw_tables, page_texts

def _continues(prev, t):
    """Is table t the continuation of prev on the next page?"""
    if prev is None or t["page"] != prev["last_page"] + 1:
        return False
    if len(t["rows"][0]) != len(prev["header"]) or t["caption"]:
        return False
    same_header = t["rows"][0] == prev["header"]  # header repeated
    at_top = t["bbox"][1] < 120                   # starts near top
    prev_at_bottom = prev["last_bottom"] > t["page_height"] - 120
    return same_header or (at_top and prev_at_bottom)

def merge_multipage(raw_tables):
    merged = []
    for t in raw_tables:
        prev = merged[-1] if merged else None
        if _continues(prev, t):
            body = t["rows"][1:] if t["rows"][0] == prev["header"] else t["rows"]
            prev["rows"].extend(body)
            prev["pages"].append(t["page"])
            prev["last_page"], prev["last_bottom"] = t["page"], t["bbox"][3]
        else:
            merged.append({
                "header": t["rows"][0],
                "rows": t["rows"][1:],
                "caption": t["caption"],
                "pages": [t["page"]],
                "last_page": t["page"],
                "last_bottom": t["bbox"][3],
            })
    return merged
