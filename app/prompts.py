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
