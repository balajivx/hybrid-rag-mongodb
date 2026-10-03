import re
from app.db import tables_meta, table_rows

SCORE = re.compile(r"^(\d+)\s*[–-]\s*(\d+)$")

def snake(s):
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower() or "col"

def unique(names):
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out

def cast(v):
    if v in ("", "—", "-"):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    return v  # ISO dates stay strings: they sort correctly

def save_tables(doc_id, merged):
    total = 0
    for i, t in enumerate(merged, start=1):
        table_id = f"table_{i}"
        cols = unique([snake(h) for h in t["header"]])
        docs = []
        for ri, r in enumerate(t["rows"]):
            if r == t["header"]:
                continue
            data = {c: cast(v) for c, v in zip(cols, r)}
            for c, v in list(data.items()):  # "3–1" -> score_1 / score_2
                m = SCORE.match(v) if isinstance(v, str) else None
                if m:
                    data[f"{c}_1"], data[f"{c}_2"] = int(m[1]), int(m[2])
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

        columns = []
        for c in docs[0]["data"]:
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
            "title": t["caption"] or f"Table starting on page {t['pages'][0]}",
            "pages": t["pages"],
            "row_count": len(docs),
            "columns": columns,
        })
    return total
