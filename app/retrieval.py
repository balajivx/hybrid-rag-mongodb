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

def vector_search(collection, question, doc_id, k=6, api_key=None):
    qv = embed([question], task="RETRIEVAL_QUERY", api_key=api_key)[0]
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
