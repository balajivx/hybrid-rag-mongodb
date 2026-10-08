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

def save_chunks(doc_id, blocks, collection, api_key=None):
    chunks = chunk_text(blocks)
    if not chunks:
        return 0
    vectors = embed([c["text"] for c in chunks], api_key=api_key)
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
