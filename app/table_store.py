import re
from app.db import tables_meta, table_rows

SCORE = re.compile(r"^(\d+)\s*[–-]\s*(\d+)$")

def snake(s):
    if s is None:
        return "col"
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()
    return cleaned or "col"

def unique(names):
    seen, out = {}, []
    for n in names:
        n = n or "col"
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out

def cast(v):
    if v is None or v in ("", "—", "-", "N/A", "n/a", "null", "None"):
        return None
    if isinstance(v, (int, float)):
        return v
    v_str = str(v).strip().replace(",", "")
    if re.fullmatch(r"-?\d+", v_str):
        try:
            return int(v_str)
        except ValueError:
            return v_str
    if re.fullmatch(r"-?\d+\.\d+", v_str):
        try:
            return float(v_str)
        except ValueError:
            return v_str
    return str(v).strip()

def save_tables(doc_id, merged):
    total = 0
    if not merged:
        return 0
    for i, t in enumerate(merged, start=1):
        table_id = f"table_{i}"
        header = t.get("header") or []
        rows = t.get("rows") or []
        if not header and rows:
            header = rows[0]
            rows = rows[1:]
        if not header:
            continue

        cols = unique([snake(h) for h in header])
        docs = []
        for ri, r in enumerate(rows):
            if r == header:
                continue
            data = {}
            for ci, c in enumerate(cols):
                val = r[ci] if ci < len(r) else None
                data[c] = cast(val)
            for c, v in list(data.items()):  # "3–1" -> score_1 / score_2
                m = SCORE.match(v) if isinstance(v, str) else None
                if m:
                    try:
                        data[f"{c}_1"], data[f"{c}_2"] = int(m[1]), int(m[2])
                    except Exception:
                        pass
            docs.append({
                "doc_id": doc_id,
                "table_id": table_id,
                "row_index": ri,
                "data": data
            })
        if not docs:
            continue
        table_rows.insert_many(docs)
        total += len(docs)

        # Collect all column names across all rows
        all_cols = []
        for d in docs:
            for k in d.get("data", {}).keys():
                if k not in all_cols:
                    all_cols.append(k)

        columns = []
        for c in all_cols:
            vals = [d["data"].get(c) for d in docs if d["data"].get(c) is not None]
            is_num = vals and all(isinstance(v, (int, float)) for v in vals)
            distinct = sorted({str(v) for v in vals})
            columns.append({
                "name": c,
                "type": "number" if is_num else "string",
                "examples": distinct[:8],
                "n_distinct": len(distinct)
            })
        tables_meta.insert_one({
            "doc_id": doc_id,
            "table_id": table_id,
            "title": t.get("caption") or f"Table starting on page {t.get('pages', [1])[0]}",
            "pages": t.get("pages", [1]),
            "row_count": len(docs),
            "columns": columns,
        })
    return total
