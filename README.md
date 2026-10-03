# ⚡ Hybrid RAG on MongoDB: Tables + Text

A next-generation multi-agent Hybrid Retrieval-Augmented Generation (RAG) system that seamlessly answers questions over complex PDF documents containing both unstructured narrative text and structured multi-page tables, using **MongoDB Atlas** as the unified operational, vector, and relational data store.

---

## 🌟 Features

- **Dual-Path Ingestion Pipeline**:
  - Automatically parses text and tables using `pdfplumber`.
  - **Multi-page table continuation merger**: Intelligently stitches long tables that span multiple pages into unified structured entities.
  - Column snake_case normalization, data type detection, score pair extraction (`score_1`, `score_2`), and automatic schema cataloging.
  - Sentence-boundary narrative chunking with Google Gemini embeddings (`gemini-embedding-2`, 768 dimensions).
- **One Database, Two Brains (MongoDB Atlas)**:
  - **Atlas Vector Search (`$vectorSearch`)** for semantic narrative retrieval.
  - **Guarded Aggregation Pipelines (`$match`, `$group`, `$sort`, etc.)** for high-precision analytical calculations and table filtering.
  - Strict security guardrails: Whitelisted read-only stages and blacklisted operators (`$where`, `$out`, `$merge`, `$lookup`, `$function`).
- **LangGraph Multi-Agent Orchestration**:
  - **Router Agent**: Dynamically classifies user queries into `text`, `table`, or `both`.
  - **RAG Agent**: Retrieves relevant narrative context chunks.
  - **Table Agent**: Generates and executes MongoDB aggregation pipelines with iterative error-correction and filter-relaxation loops.
  - **Combiner Node**: Synthesizes the final answer with page citations `(p. N)` and table references.
- **Knowledge Vault & Multi-Document Support**:
  - Batch upload multiple PDFs.
  - Interactive document management and schema inspector.
- **Side-by-Side Evaluation**:
  - Real-time comparison between Conventional Baseline RAG and Agentic Hybrid RAG.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- MongoDB Atlas cluster (M0 free tier supported)
- Google Gemini API key

### 2. Installation
```bash
git clone https://github.com/balajivx/hybrid-rag-mongodb.git
cd hybrid-rag-mongodb

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
DB_NAME=hybrid_rag
GEMINI_API_KEY=<your-gemini-api-key>
LLM_MODEL=gemini-2.5-flash
EMBED_MODEL=gemini-embedding-2
EMBED_DIM=768
API_URL=http://localhost:8000
```

### 4. Create Search Indexes
```bash
python -m scripts.create_indexes
```

### 5. Launch Backend & UI
In one terminal, start the FastAPI backend:
```bash
uvicorn app.api:api --host 0.0.0.0 --port 8000
```

In a second terminal, start the Streamlit UI:
```bash
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧪 Testing & Evaluation

Run the automated test suite:
```bash
pytest -v
```

Run the benchmark evaluation against an ingested document:
```bash
python -m scripts.run_eval <doc_id>
```

---

## 📜 License
CC-BY-SA 4.0 / MIT
