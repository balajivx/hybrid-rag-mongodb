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
    print(f"Warning: {coll.name} index not queryable after {timeout}s (it may still be building on Atlas).")

if __name__ == "__main__":
    ensure_indexes()
    for c in (text_chunks, baseline_chunks):
        create_vector_index(c)
        wait_until_queryable(c)
