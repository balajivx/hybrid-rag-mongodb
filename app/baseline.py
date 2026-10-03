from app import prompts
from app.db import baseline_chunks
from app.llm import generate
from app.retrieval import vector_search

def baseline_answer(question, doc_id, k=6):
    hits = vector_search(baseline_chunks, question, doc_id, k=k)
    context = "\n\n".join(f'(p. {h["page"]}) {h["text"]}' for h in hits)
    answer = generate(prompts.BASELINE.format(context=context, question=question))
    return {"answer": answer, "chunks": hits}
