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
    api_key: str

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
    out = generate(
        prompts.ROUTER.format(catalog=catalog, question=state["question"]),
        json_mode=True,
        api_key=state.get("api_key")
    )
    route = out.get("route", "both")
    return {
        "route": route if route in ("table", "text", "both") else "both",
        "route_reason": out.get("reason", "")
    }

def rag_node(state):
    hits = vector_search(text_chunks, state["question"], state["doc_id"], k=6, api_key=state.get("api_key"))
    return {"text_result": hits}

def table_node(state, max_attempts=3):
    catalog = format_catalog(table_catalog(state["doc_id"]))
    context = ""
    if state.get("text_result"):  # "both": text informs the query
        ctx = "\n".join(h["text"] for h in state["text_result"])[:3000]
        context = f"\nContext from the narrative (use it to resolve names):\n{ctx}\n"
    feedback, attempts = "", []
    for _ in range(max_attempts):
        plan = generate(
            prompts.TABLE.format(
                catalog=catalog,
                context=context,
                feedback=feedback,
                question=state["question"]
            ),
            json_mode=True,
            api_key=state.get("api_key")
        )
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
    answer = generate(
        prompts.COMBINER.format(
            question=state["question"],
            table_evidence=table_ev,
            text_evidence=text_ev
        ),
        api_key=state.get("api_key")
    )
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
