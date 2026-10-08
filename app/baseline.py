from app import prompts
from app.db import baseline_chunks
from app.llm import generate
from app.retrieval import vector_search

def baseline_answer(question, doc_id, k=6, api_key=None):
    hits = vector_search(baseline_chunks, question, doc_id, k=k, api_key=api_key)
    context = "\n\n".join(f'(p. {h["page"]}) {h["text"]}' for h in hits)
    answer = generate(prompts.BASELINE.format(context=context, question=question), api_key=api_key)
    return {"answer": answer, "chunks": hits}
