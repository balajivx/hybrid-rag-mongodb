import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from app.db import documents, tables_meta, table_rows, text_chunks, baseline_chunks
from app.ingest import ingest_pdf
from app.agents import hybrid_app
from app.baseline import baseline_answer
from app.retrieval import table_catalog

api = FastAPI(title="Hybrid RAG on MongoDB")

class AskRequest(BaseModel):
    doc_id: str
    question: str

@api.get("/documents")
def list_documents():
    docs = list(documents.find({}, {"_id": 1, "filename": 1, "tables": 1, "table_rows": 1, "text_chunks": 1, "baseline_chunks": 1, "ingested_at": 1}))
    return [
        {
            "doc_id": d["_id"],
            "filename": d.get("filename", "Unknown"),
            "tables": d.get("tables", 0),
            "table_rows": d.get("table_rows", 0),
            "text_chunks": d.get("text_chunks", 0),
            "baseline_chunks": d.get("baseline_chunks", 0),
            "ingested_at": str(d.get("ingested_at", ""))
        }
        for d in docs
    ]

@api.get("/documents/{doc_id}/tables")
def get_document_tables(doc_id: str):
    return table_catalog(doc_id)

@api.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    for coll in (documents, tables_meta, table_rows, text_chunks, baseline_chunks):
        if coll == documents:
            coll.delete_one({"_id": doc_id})
        else:
            coll.delete_many({"doc_id": doc_id})
    return {"status": "deleted", "doc_id": doc_id}

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

@api.post("/ingest-multiple")
def ingest_multiple(files: List[UploadFile] = File(...)):
    summaries = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(f.file, tmp)
            path = tmp.name
        try:
            summary = ingest_pdf(path, f.filename)
            summaries.append(summary)
        finally:
            if os.path.exists(path):
                os.remove(path)
    return summaries

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
