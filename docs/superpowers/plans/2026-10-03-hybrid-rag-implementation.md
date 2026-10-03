# Hybrid RAG on MongoDB Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete end-to-end multi-agent Hybrid RAG application on MongoDB Atlas with dual table/text processing, Google Gemini embeddings & LLM, LangGraph orchestration, FastAPI REST service, and Streamlit demo UI.

**Architecture:** A dual-path RAG system splitting documents into narrative text chunks ($vectorSearch) and structured multi-page tables (MongoDB aggregation pipelines). A LangGraph router directs queries to text, table, or both (with text context passing to table aggregation for multi-hop queries), with a grounded combiner producing final citations.

**Tech Stack:** Python 3.10+, MongoDB Atlas, `pymongo`, `google-genai`, `langgraph`, `pdfplumber`, `fastapi`, `uvicorn`, `streamlit`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-03-hybrid-rag-design.md`

## Global Constraints
- Python 3.10+
- Models configurable via `.env`: `LLM_MODEL=gemini-2.5-flash` (or user model), `EMBED_MODEL=gemini-embedding-exp-03-07` / `text-embedding-004`, `EMBED_DIM=768`
- Unified MongoDB Atlas database for text chunks, baseline chunks, table rows, metadata catalog, and documents
- Strict aggregation stage whitelist (`$match`, `$group`, `$project`, `$sort`, `$limit`, `$count`, `$unwind`, `$addFields`, `$set`, `$unset`, `$skip`) and operator blacklist (`$where`, `$function`, `$accumulator`, `$out`, `$merge`, `$lookup`, `$unionWith`)

---

### Task 1: Environment, Dependencies & Configuration

**Files:**
- Create: `requirements.txt`
- Create: `.env`
- Create: `.gitignore`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: Environment variables (`MONGODB_URI`, `GEMINI_API_KEY`, `DB_NAME`, `LLM_MODEL`, `EMBED_MODEL`, `EMBED_DIM`)
- Produces: `app.config` constants (`MONGODB_URI`, `DB_NAME`, `GEMINI_API_KEY`, `LLM_MODEL`, `EMBED_MODEL`, `EMBED_DIM`, `VECTOR_INDEX`, `MIN_TABLE_ROWS`, `API_URL`)

- [ ] **Step 1: Write test for config loading**
```python
# tests/test_config.py
def test_config_variables():
    from app import config
    assert config.DB_NAME == "hybrid_rag"
    assert config.VECTOR_INDEX == "chunk_vector_index"
    assert config.MIN_TABLE_ROWS == 8
    assert config.EMBED_DIM == 768
    assert bool(config.MONGODB_URI) is True
    assert bool(config.GEMINI_API_KEY) is True
```

- [ ] **Step 2: Create .gitignore and requirements.txt**
```text
# requirements.txt
pymongo>=4.7
google-genai>=1.0
langgraph>=0.2
pdfplumber>=0.11
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
streamlit>=1.35
requests>=2.31
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 3: Create .env with provided credentials and install dependencies**
```text
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
DB_NAME=hybrid_rag
GEMINI_API_KEY=<your-gemini-api-key>
LLM_MODEL=gemini-2.5-flash
EMBED_MODEL=gemini-embedding-exp-03-07
EMBED_DIM=768
API_URL=http://localhost:8000
```

- [ ] **Step 4: Implement app/config.py**
```python
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.getenv("DB_NAME", "hybrid_rag")
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-exp-03-07")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
VECTOR_INDEX = "chunk_vector_index"
MIN_TABLE_ROWS = 8  # smaller grids are treated as text
API_URL = os.getenv("API_URL", "http://localhost:8000")
```

- [ ] **Step 5: Run test to verify config passes**
Run `pytest tests/test_config.py`

---

### Task 2: Database and LLM Helpers

**Files:**
- Create: `app/db.py`
- Create: `app/llm.py`
- Test: `tests/test_db_llm.py`

**Interfaces:**
- Consumes: `app.config`
- Produces: `app.db` (`client`, `db`, `documents`, `tables_meta`, `table_rows`, `text_chunks`, `baseline_chunks`, `ensure_indexes()`), `app.llm.embed(texts, task, batch)`, `app.llm.generate(prompt, json_mode)`

- [ ] **Step 1: Write test for db indexes and llm mock / connectivity**
```python
# tests/test_db_llm.py
from app.db import db, ensure_indexes, documents, tables_meta, table_rows, text_chunks, baseline_chunks
from app.llm import embed, generate

def test_db_collections():
    assert documents.name == "documents"
    assert tables_meta.name == "tables_meta"
    assert table_rows.name == "table_rows"
    assert text_chunks.name == "text_chunks"
    assert baseline_chunks.name == "baseline_chunks"
```

- [ ] **Step 2: Implement app/db.py**
```python
from pymongo import MongoClient, ASCENDING
from app.config import MONGODB_URI, DB_NAME

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
```

- [ ] **Step 3: Implement app/llm.py**
```python
import json
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, LLM_MODEL, EMBED_MODEL, EMBED_DIM

client = genai.Client(api_key=GEMINI_API_KEY)

def embed(texts, task="RETRIEVAL_DOCUMENT", batch=50):
    """Embed a list of strings. Use task='RETRIEVAL_QUERY' for questions."""
    if not texts:
        return []
    vectors = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i + batch]
        try:
            res = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch_texts,
                config=types.EmbedContentConfig(
                    task_type=task,
                    output_dimensionality=EMBED_DIM
                ),
            )
            vectors.extend(e.values for e in res.embeddings)
        except Exception:
            # Fallback if task_type / output_dimensionality not supported on specific model
            res = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch_texts,
            )
            vectors.extend(e.values for e in res.embeddings)
    return vectors

def generate(prompt, json_mode=False):
    cfg = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=cfg
    )
    return json.loads(resp.text) if json_mode else resp.text
```

- [ ] **Step 4: Run test to verify passes**
Run `pytest tests/test_db_llm.py`

---

### Task 3: PDF Parsing & Multi-Page Table Merging

**Files:**
- Create: `app/pdf_parser.py`
- Test: `tests/test_pdf_parser.py`

**Interfaces:**
- Consumes: PDF file path
- Produces: `parse_pdf(path) -> (text_blocks, raw_tables, page_texts)`, `merge_multipage(raw_tables) -> merged_tables`

- [ ] **Step 1: Write test for table merging logic**
```python
# tests/test_pdf_parser.py
from app.pdf_parser import merge_multipage

def test_merge_multipage_continuation():
    raw_tables = [
        {
            "page": 1, "bbox": (50, 200, 500, 750), "page_height": 800,
            "rows": [["Col1", "Col2"], ["Val1", "Val2"]],
            "caption": "Table 1: Example"
        },
        {
            "page": 2, "bbox": (50, 50, 500, 700), "page_height": 800,
            "rows": [["Col1", "Col2"], ["Val3", "Val4"]],
            "caption": None
        }
    ]
    merged = merge_multipage(raw_tables)
    assert len(merged) == 1
    assert merged[0]["caption"] == "Table 1: Example"
    assert merged[0]["pages"] == [1, 2]
    assert len(merged[0]["rows"]) == 2  # ["Val1", "Val2"], ["Val3", "Val4"]
```

- [ ] **Step 2: Implement app/pdf_parser.py**
```python
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
```

- [ ] **Step 3: Run tests to verify**
Run `pytest tests/test_pdf_parser.py`

---

### Task 4: Table Store & Text Store

**Files:**
- Create: `app/table_store.py`
- Create: `app/text_store.py`
- Test: `tests/test_stores.py`

**Interfaces:**
- Consumes: Parsed blocks / tables, `app.db`, `app.llm.embed`
- Produces: `save_tables(doc_id, merged) -> int`, `chunk_text(blocks, size, overlap) -> list`, `save_chunks(doc_id, blocks, collection) -> int`

- [ ] **Step 1: Write test for snake_case, type cast, score splitting, chunking**
```python
# tests/test_stores.py
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
```

- [ ] **Step 2: Implement app/table_store.py**
```python
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
```

- [ ] **Step 3: Implement app/text_store.py**
```python
import re
from app.llm import embed

def chunk_text(blocks, size=1200, overlap=200):
    chunks = []
    for b in blocks:
        text = re.sub(r"\s+", " ", b["text"]).strip()
        current = ""
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if current and len(current) + len(sent) > size:
                chunks.append({"page": b["page"], "text": current.strip()})
                current = current[-overlap:]
            current += " " + sent
        if current.strip():
            chunks.append({"page": b["page"], "text": current.strip()})
    return chunks

def save_chunks(doc_id, blocks, collection):
    chunks = chunk_text(blocks)
    if not chunks:
        return 0
    vectors = embed([c["text"] for c in chunks])
    docs = [
        {
            "doc_id": doc_id,
            "chunk_no": i,
            "page": c["page"],
            "text": c["text"],
            "embedding": v
        }
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    if docs:
        collection.insert_many(docs)
    return len(docs)
```

- [ ] **Step 4: Run tests**
Run `pytest tests/test_stores.py`

---

### Task 5: Ingestion Pipeline

**Files:**
- Create: `app/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: PDF path, filename
- Produces: `ingest_pdf(path, filename) -> summary dict`

- [ ] **Step 1: Write mock test for ingest_pdf**
- [ ] **Step 2: Implement app/ingest.py**
```python
import hashlib
from datetime import datetime, timezone
from app.db import documents, tables_meta, table_rows, text_chunks, baseline_chunks
from app.pdf_parser import parse_pdf, merge_multipage
from app.table_store import save_tables
from app.text_store import save_chunks

def ingest_pdf(path, filename="document.pdf"):
    with open(path, "rb") as f:
        doc_id = hashlib.sha1(f.read()).hexdigest()[:16]

    for coll in (tables_meta, table_rows, text_chunks, baseline_chunks):
        coll.delete_many({"doc_id": doc_id})  # idempotent re-ingestion

    text_blocks, raw_tables, page_texts = parse_pdf(path)
    merged = merge_multipage(raw_tables)
    n_rows = save_tables(doc_id, merged)
    n_chunks = save_chunks(doc_id, text_blocks, text_chunks)
    n_base = save_chunks(doc_id, page_texts, baseline_chunks)

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
```

- [ ] **Step 3: Run test**
Run `pytest tests/test_ingest.py`

---

### Task 6: Index Creation Script

**Files:**
- Create: `scripts/create_indexes.py`

**Interfaces:**
- Consumes: `app.db`, `app.config.VECTOR_INDEX`, `app.config.EMBED_DIM`
- Produces: Vector search indexes on `text_chunks` and `baseline_chunks`, waiting until `queryable`

- [ ] **Step 1: Implement scripts/create_indexes.py**
```python
import time
from pymongo.operations import SearchIndexModel
from app.db import db, text_chunks, baseline_chunks, ensure_indexes
from app.config import EMBED_DIM, VECTOR_INDEX

def create_vector_index(coll):
    if coll.name not in db.list_collection_names():
        db.create_collection(coll.name)
    existing = list(coll.list_search_indexes())
    if any(ix["name"] == VECTOR_INDEX for ix in existing):
        print(f"{coll.name}: search index '{VECTOR_INDEX}' already exists")
        return
    print(f"{coll.name}: creating search index '{VECTOR_INDEX}'...")
    coll.create_search_index(SearchIndexModel(
        name=VECTOR_INDEX,
        type="vectorSearch",
        definition={"fields": [
            {"type": "vector", "path": "embedding",
             "numDimensions": EMBED_DIM, "similarity": "cosine"},
            {"type": "filter", "path": "doc_id"},
        ]},
    ))

def wait_until_queryable(coll, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        ix = next(iter(coll.list_search_indexes(VECTOR_INDEX)), None)
        if ix and ix.get("queryable"):
            print(f"{coll.name}: index ready")
            return
        time.sleep(5)
    print(f"Warning: {coll.name} index not ready after {timeout}s (it may still be building on Atlas).")

if __name__ == "__main__":
    ensure_indexes()
    for c in (text_chunks, baseline_chunks):
        create_vector_index(c)
        wait_until_queryable(c)
```

- [ ] **Step 2: Run script to initialize MongoDB indexes**
Run `python -m scripts.create_indexes`

---

### Task 7: Retrieval Module (Vector Search & Guarded Aggregations)

**Files:**
- Create: `app/retrieval.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: collections, query, doc_id, LLM pipeline
- Produces: `vector_search(collection, question, doc_id, k)`, `table_catalog(doc_id)`, `run_table_pipeline(doc_id, table_id, pipeline, max_rows)`

- [ ] **Step 1: Write tests for stage validation and injection prevention**
```python
# tests/test_retrieval.py
import pytest
from app.retrieval import run_table_pipeline

def test_blocked_stages():
    with pytest.raises(ValueError):
        run_table_pipeline("doc1", "table_1", [{"$out": "other_coll"}])
    with pytest.raises(ValueError):
        run_table_pipeline("doc1", "table_1", [{"$where": "this.a == 1"}])
```

- [ ] **Step 2: Implement app/retrieval.py**
```python
from bson import ObjectId
from app.db import table_rows, tables_meta
from app.llm import embed
from app.config import VECTOR_INDEX

ALLOWED_STAGES = {
    "$match", "$group", "$project", "$sort", "$limit", "$count",
    "$unwind", "$addFields", "$set", "$unset", "$skip"
}
BLOCKED_OPERATORS = {
    "$where", "$function", "$accumulator", "$out", "$merge",
    "$lookup", "$unionWith"
}

def vector_search(collection, question, doc_id, k=6):
    qv = embed([question], task="RETRIEVAL_QUERY")[0]
    pipeline = [
        {"$vectorSearch": {
            "index": VECTOR_INDEX, "path": "embedding", "queryVector": qv,
            "numCandidates": 150, "limit": k, "filter": {"doc_id": doc_id},
        }},
        {"$project": {
            "_id": 0, "text": 1, "page": 1,
            "score": {"$meta": "vectorSearchScore"}
        }},
    ]
    return list(collection.aggregate(pipeline))

def table_catalog(doc_id):
    return list(tables_meta.find({"doc_id": doc_id}, {"_id": 0, "doc_id": 0}))

def _walk(node):
    if isinstance(node, dict):
        for key, val in node.items():
            if key in BLOCKED_OPERATORS:
                raise ValueError(f"Operator {key} is not allowed")
            _walk(val)
    elif isinstance(node, list):
        for val in node:
            _walk(val)

def _jsonable(node):
    if isinstance(node, ObjectId):
        return str(node)
    if isinstance(node, dict):
        return {k: _jsonable(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_jsonable(v) for v in node]
    return node

def run_table_pipeline(doc_id, table_id, pipeline, max_rows=200):
    """Validate an LLM-written pipeline, scope it to one table and run it."""
    if not isinstance(pipeline, list):
        raise ValueError("pipeline must be a list of stages")
    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1 or next(iter(stage)) not in ALLOWED_STAGES:
            raise ValueError(f"Stage not allowed: {stage}")
    _walk(pipeline)
    scoped = ([{"$match": {"doc_id": doc_id, "table_id": table_id}}]
              + pipeline + [{"$limit": max_rows}])
    return _jsonable(list(table_rows.aggregate(scoped, maxTimeMS=5000)))
```

- [ ] **Step 3: Run test**
Run `pytest tests/test_retrieval.py`

---

### Task 8: Prompts & Conventional Baseline RAG

**Files:**
- Create: `app/prompts.py`
- Create: `app/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Produces: `prompts.ROUTER`, `prompts.TABLE`, `prompts.COMBINER`, `prompts.BASELINE`, `baseline_answer(question, doc_id, k)`

- [ ] **Step 1: Implement app/prompts.py**
```python
ROUTER = """You route questions about one PDF document.
The document has narrative text and these structured tables:
{catalog}

Choose the source(s) needed to answer:
- "table": filtering, counting, ranking, aggregating or looking up rows in a table.
- "text": explanations, stories, descriptions or facts that are not in any table.
- "both": the answer needs a fact from the narrative AND data from a table
  (for example, the narrative names a player or team and the table holds the numbers).

Return JSON: {{"route": "table" | "text" | "both", "reason": "<one sentence>"}}
Question: {question}"""

TABLE = """You write MongoDB aggregation pipelines.
Each document in the collection is one table row:
  {{"table_id": "...", "row_index": 0, "data": {{"<column>": <value>, ...}}}}
Refer to columns as "data.<column>". Your pipeline runs AFTER an automatic
{{"$match": {{"table_id": "<the table you choose>"}}}}.

Tables (columns, types and example values):
{catalog}

Rules:
- Use only: $match, $group, $project, $sort, $limit, $count, $unwind,
  $addFields, $set, $unset, $skip.
- Text values may contain accents; when unsure of spelling use
  {{"$regex": "<pattern>", "$options": "i"}}.
- Dates are ISO strings (YYYY-MM-DD); compare them as strings.
- Score columns like "3–1" also have numeric <column>_1 and <column>_2 fields.
- Project only the fields needed to answer.
{context}{feedback}
Return JSON: {{"table_id": "...", "pipeline": [ ... ], "explanation": "..."}}
Question: {question}"""

COMBINER = """Answer the question using only the evidence below.
- Use table evidence for numbers and lists; use text evidence for context.
- If the evidence is insufficient, say exactly what is missing.
- Cite text evidence as (p. N) and table evidence by the table title.

Question: {question}

Table evidence:
{table_evidence}

Text evidence:
{text_evidence}

Answer:"""

BASELINE = """Answer the question using only the context below.
If the context does not contain the answer, say so.

Context:
{context}

Question: {question}
Answer:"""
```

- [ ] **Step 2: Implement app/baseline.py**
```python
from app import prompts
from app.db import baseline_chunks
from app.llm import generate
from app.retrieval import vector_search

def baseline_answer(question, doc_id, k=6):
    hits = vector_search(baseline_chunks, question, doc_id, k=k)
    context = "\n\n".join(f'(p. {h["page"]}) {h["text"]}' for h in hits)
    answer = generate(prompts.BASELINE.format(context=context, question=question))
    return {"answer": answer, "chunks": hits}
```

---

### Task 9: LangGraph Multi-Agent Orchestration

**Files:**
- Create: `app/agents.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `app.prompts`, `app.db`, `app.llm`, `app.retrieval`
- Produces: `hybrid_app` (compiled LangGraph graph)

- [ ] **Step 1: Implement app/agents.py**
```python
import json
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from app import prompts
from app.db import text_chunks
from app.llm import generate
from app.retrieval import vector_search, table_catalog, run_table_pipeline

class QAState(TypedDict, total=False):
    question: str
    doc_id: str
    route: str
    route_reason: str
    text_result: list
    table_result: dict
    answer: str

def format_catalog(catalog):
    lines = []
    for t in catalog:
        cols = "; ".join(f'{c["name"]} ({c["type"]}, e.g. {c["examples"][:4]})'
                         for c in t["columns"])
        lines.append(f'- {t["table_id"]}: "{t["title"]}", {t["row_count"]} rows. '
                     f'Columns: {cols}')
    return "\n".join(lines) or "(no tables)"

def router_node(state):
    catalog = format_catalog(table_catalog(state["doc_id"]))
    out = generate(prompts.ROUTER.format(catalog=catalog,
                                         question=state["question"]), json_mode=True)
    route = out.get("route", "both")
    return {
        "route": route if route in ("table", "text", "both") else "both",
        "route_reason": out.get("reason", "")
    }

def rag_node(state):
    hits = vector_search(text_chunks, state["question"], state["doc_id"], k=6)
    return {"text_result": hits}

def table_node(state, max_attempts=3):
    catalog = format_catalog(table_catalog(state["doc_id"]))
    context = ""
    if state.get("text_result"):  # "both": text informs the query
        ctx = "\n".join(h["text"] for h in state["text_result"])[:3000]
        context = f"\nContext from the narrative (use it to resolve names):\n{ctx}\n"
    feedback, attempts = "", []
    for _ in range(max_attempts):
        plan = generate(prompts.TABLE.format(catalog=catalog, context=context,
                                             feedback=feedback,
                                             question=state["question"]),
                        json_mode=True)
        try:
            rows = run_table_pipeline(state["doc_id"], plan["table_id"], plan["pipeline"])
            attempts.append({"plan": plan, "rows": len(rows)})
            if rows:
                return {"table_result": {**plan, "rows": rows, "attempts": attempts}}
            feedback = ("\nYour previous pipeline returned no rows: "
                        f"{json.dumps(plan['pipeline'])}. Relax the filters.\n")
        except Exception as exc:  # invalid stage, bad JSON, timeout
            attempts.append({"plan": plan, "error": str(exc)})
            feedback = f"\nYour previous pipeline failed with: {exc}. Fix it.\n"
    return {"table_result": {"rows": [], "attempts": attempts}}

def combiner_node(state):
    tr = state.get("table_result")
    table_ev = "(not used)"
    if tr:
        table_ev = json.dumps({
            "table_id": tr.get("table_id"),
            "pipeline": tr.get("pipeline"),
            "rows": tr.get("rows", [])[:50]
        }, ensure_ascii=False, default=str)
    text_ev = "\n\n".join(f'(p. {h["page"]}) {h["text"]}'
                          for h in state.get("text_result", [])) or "(not used)"
    answer = generate(prompts.COMBINER.format(
        question=state["question"],
        table_evidence=table_ev,
        text_evidence=text_ev
    ))
    return {"answer": answer}

def after_router(state):
    return "table_agent" if state["route"] == "table" else "rag_agent"

def after_rag(state):
    return "table_agent" if state["route"] == "both" else "combiner"

graph = StateGraph(QAState)
graph.add_node("router", router_node)
graph.add_node("rag_agent", rag_node)
graph.add_node("table_agent", table_node)
graph.add_node("combiner", combiner_node)
graph.add_edge(START, "router")
graph.add_conditional_edges("router", after_router, ["table_agent", "rag_agent"])
graph.add_conditional_edges("rag_agent", after_rag, ["table_agent", "combiner"])
graph.add_edge("table_agent", "combiner")
graph.add_edge("combiner", END)

hybrid_app = graph.compile()
```

---

### Task 10: FastAPI Service

**Files:**
- Create: `app/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: FastAPI app with `/health`, `/ingest`, `/ask/hybrid`, `/ask/baseline`

- [ ] **Step 1: Implement app/api.py**
```python
import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from app.ingest import ingest_pdf
from app.agents import hybrid_app
from app.baseline import baseline_answer

api = FastAPI(title="Hybrid RAG on MongoDB")

class AskRequest(BaseModel):
    doc_id: str
    question: str

@api.post("/ingest")
def ingest(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = tmp.name
    try:
        return ingest_pdf(path, file.filename)
    finally:
        if os.path.exists(path):
            os.remove(path)

@api.post("/ask/hybrid")
def ask_hybrid(req: AskRequest):
    s = hybrid_app.invoke({"question": req.question, "doc_id": req.doc_id})
    return {
        "answer": s["answer"],
        "route": s["route"],
        "route_reason": s.get("route_reason"),
        "table_result": s.get("table_result"),
        "text_result": s.get("text_result", [])
    }

@api.post("/ask/baseline")
def ask_baseline(req: AskRequest):
    return baseline_answer(req.question, req.doc_id)

@api.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Test FastAPI app**
Run `pytest tests/test_api.py`

---

### Task 11: Streamlit UI

**Files:**
- Create: `ui/streamlit_app.py`

**Interfaces:**
- Produces: Side-by-side interactive web interface connecting to FastAPI backend

- [ ] **Step 1: Implement ui/streamlit_app.py**
```python
import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Hybrid RAG on MongoDB", layout="wide")
st.title("Hybrid RAG on MongoDB: tables + text")

pdf = st.file_uploader("Upload a PDF", type="pdf")
if pdf and st.button("Ingest"):
    with st.spinner("Parsing, embedding and storing in MongoDB..."):
        r = requests.post(
            f"{API}/ingest",
            timeout=900,
            files={"file": (pdf.name, pdf.getvalue(), "application/pdf")}
        )
        r.raise_for_status()
        st.session_state["doc"] = r.json()

doc = st.session_state.get("doc")
if doc:
    st.success(f"{doc['filename']}: {doc['tables']} tables, {doc['table_rows']} rows, "
               f"{doc['text_chunks']} text chunks")
    question = st.text_input("Ask a question about the document")
    if question:
        body = {"doc_id": doc["doc_id"], "question": question}
        left, right = st.columns(2)
        with left:
            st.subheader("Conventional RAG")
            with st.spinner("Querying baseline RAG..."):
                b = requests.post(f"{API}/ask/baseline", json=body, timeout=300).json()
            st.write(b["answer"])
            with st.expander("Retrieved chunks"):
                st.json(b["chunks"])
        with right:
            st.subheader("Hybrid RAG")
            with st.spinner("Routing & querying Hybrid RAG..."):
                h = requests.post(f"{API}/ask/hybrid", json=body, timeout=300).json()
            st.write(h["answer"])
            st.caption(f"Route: {h.get('route')} - {h.get('route_reason')}")
            if h.get("table_result"):
                with st.expander("Aggregation pipeline and rows"):
                    st.json(h["table_result"])
            if h.get("text_result"):
                with st.expander("Text chunks"):
                    st.json(h["text_result"])
```

---

### Task 12: Evaluation Suite & Benchmark

**Files:**
- Create: `tests/eval_questions.json`
- Create: `scripts/run_eval.py`

**Interfaces:**
- Produces: Evaluation dataset & accuracy scoring script

- [ ] **Step 1: Implement tests/eval_questions.json**
- [ ] **Step 2: Implement scripts/run_eval.py**

---

### Task 13: End-to-End Verification & Launch Verification

**Interfaces:**
- Verifies full pipeline, index creation on Atlas, ingestion, querying, and UI readiness.
