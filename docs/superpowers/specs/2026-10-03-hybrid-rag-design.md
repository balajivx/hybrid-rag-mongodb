# Hybrid RAG on MongoDB: Specification & Design Document

## 1. Overview
This project implements a multi-agent Hybrid Retrieval-Augmented Generation (RAG) system using MongoDB Atlas as a unified operational, vector, and structured data store. The system handles unstructured narrative prose alongside structured multi-page tabular data from complex PDF documents (such as sports records, financial reports, or technical manuals).

## 2. Architecture & Components

```
                +----------------------------+
                |        PDF Document        |
                +--------------+-------------+
                               |
                   [ pdfplumber parser ]
                               |
              +----------------+----------------+
              |                                 |
      [ Narrative Text ]               [ Structured Tables ]
              |                                 |
   [ Sentence Chunking ]               [ Multi-page Merge ]
              |                                 |
   [ Gemini Embedding ]                [ Type-casting & Schema Catalog ]
              |                                 |
              v                                 v
   +----------------------+           +----------------------+
   | text_chunks          |           | table_rows           |
   | (Atlas Vector Index) |           | tables_meta          |
   +----------+-----------+           +----------+-----------+
              |                                  |
              +----------------+-----------------+
                               |
                    +----------v----------+
                    |  LangGraph Router   |
                    +----+-----+-----+----+
                         |     |     |
         +---------------+     |     +---------------+
         | "text"              | "both"              | "table"
         v                     v                     v
   +-----------+         +-----------+         +-----------+
   | RAG Agent |         | RAG Agent |         |Table Agent|
   +-----+-----+         +-----+-----+         +-----+-----+
         |                     |                     |
         |                     v                     |
         |               +-----------+               |
         |               |Table Agent|               |
         |               +-----+-----+               |
         |                     |                     |
         +---------------->----+<--------------------+
                               |
                               v
                     +-------------------+
                     |   Combiner Node   |
                     +---------+---------+
                               |
                               v
                     [ Grounded Answer ]
```

### 2.1 Storage Schema (MongoDB Atlas)
- **`documents`**: Ingested document metadata (`_id` (SHA1), `filename`, `counts`, `ingested_at`).
- **`tables_meta`**: Catalog per logical table (`doc_id`, `table_id`, `title`, `pages`, `row_count`, `columns[{name, type, examples, n_distinct}]`).
- **`table_rows`**: Individual row records (`doc_id`, `table_id`, `row_index`, `data: {column: value, score_1: int, score_2: int}`).
- **`text_chunks`**: Segmented narrative text (`doc_id`, `chunk_no`, `page`, `text`, `embedding` [768-dim]).
- **`baseline_chunks`**: Conventional flattened page chunks (`doc_id`, `chunk_no`, `page`, `text`, `embedding` [768-dim]).

### 2.2 Ingestion Pipeline (`app/pdf_parser.py`, `app/table_store.py`, `app/text_store.py`, `app/ingest.py`)
- Extraction of tables with row threshold (>= 8 rows considered structured tables; < 8 rows remain in narrative flow).
- Multi-page table continuation algorithm based on header matching, column counts, and top/bottom page margin positions.
- Column snake_case normalization, numerical/date type casting, and regex extraction for score pairs.
- Sentence-boundary text chunking (1,200 chars, 200 char overlap).
- Dual embedding generation using Google Gemini embeddings (`RETRIEVAL_DOCUMENT`).

### 2.3 Guarded Aggregation & Vector Retrieval (`app/retrieval.py`)
- **Vector Search**: MongoDB Atlas `$vectorSearch` with cosine similarity, filtered by `doc_id`.
- **Guarded Aggregation**: Whitelist validation allowing only safe read-only pipeline stages (`$match`, `$group`, `$project`, `$sort`, `$limit`, `$count`, `$unwind`, `$addFields`, `$set`, `$unset`, `$skip`), while rejecting blacklisted operators (`$out`, `$merge`, `$lookup`, `$where`, `$function`, `$unionWith`). Enforces scoping by `doc_id` and `table_id`, execution timeouts (`maxTimeMS=5000`), and max returned rows limit.

### 2.4 Agent Graph (`app/prompts.py`, `app/agents.py`, `app/baseline.py`)
- **Router Node**: Routes to `"text"`, `"table"`, or `"both"` using schema catalog and query context.
- **RAG Agent**: Retrieves top-k text chunks.
- **Table Agent**: Generates MongoDB aggregation pipelines with automatic retry and feedback loops (up to 3 attempts upon syntax error or empty results).
- **Combiner Node**: Synthesizes the response with clear citations `(p. N)` and table references.
- **Baseline RAG**: Conventional retrieval over flattened page text for comparative benchmarking.

### 2.5 API & UI (`app/api.py`, `ui/streamlit_app.py`)
- **FastAPI**:
  - `POST /ingest`: Accepts PDF file upload, runs parsing & embedding, returns document summary.
  - `POST /ask/hybrid`: Executes multi-agent LangGraph workflow.
  - `POST /ask/baseline`: Executes baseline vector retrieval + answer generation.
  - `GET /health`: Health check endpoint.
- **Streamlit Demo UI**: Side-by-side comparison interface showing conventional vs hybrid RAG answers, retrieved chunks, routing reasoning, and generated aggregation pipelines.

### 2.6 Evaluation & Setup Scripts (`scripts/create_indexes.py`, `scripts/run_eval.py`, `tests/eval_questions.json`)
- Programmatic `$vectorSearch` search index creation and polling for readiness.
- Evaluation test runner calculating accuracy scores across question types.

## 3. Technology Stack & Dependencies
- Python 3.10+
- `pymongo>=4.7`
- `google-genai>=1.0`
- `langgraph>=0.2`
- `pdfplumber>=0.11`
- `fastapi>=0.110`
- `uvicorn[standard]>=0.29`
- `python-multipart>=0.0.9`
- `streamlit>=1.35`
- `requests>=2.31`
- `python-dotenv>=1.0`

## 4. Verification & Testing Strategy
1. Unit and module verification for parsing, multi-page merging, and guarded aggregation execution.
2. Vector Search Index verification on MongoDB Atlas.
3. Test ingestion with sample PDF document.
4. FastAPI endpoint validation (`/health`, `/ingest`, `/ask/hybrid`, `/ask/baseline`).
5. Evaluation questions benchmark execution comparing baseline vs hybrid accuracy.
6. Streamlit UI launch and manual query testing.
