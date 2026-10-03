import os
import time
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------
# Page Configuration & Custom CSS (Inspired by anupams.in)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hybrid RAG on MongoDB | Tables + Text",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');

/* Typography - Safe selection that preserves Streamlit icon fonts */
html, body, p, h1, h2, h3, h4, h5, h6, input, textarea, button {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Explicitly preserve Streamlit Material Icons */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons, [data-testid="stExpanderToggleIcon"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}

/* Background Atmosphere */
.stApp {
    background: radial-gradient(circle at 50% 0%, #171d31 0%, #0a0d16 60%, #06080e 100%);
    color: #e2e8f0;
}

/* Clean Expander Styling */
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    margin-top: 10px !important;
    margin-bottom: 10px !important;
}

/* Glassmorphic Container Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.035);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.glass-card:hover {
    border-color: rgba(255, 255, 255, 0.16);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45);
    transform: translateY(-2px);
}

/* Hero Badge & Title */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #a5b4fc;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #ffffff 30%, #94a3b8 70%, #64748b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 780px;
    line-height: 1.5;
    margin-bottom: 24px;
}

/* Route Badges */
.badge-route-table {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: inline-block;
}
.badge-route-text {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #818cf8;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: inline-block;
}
.badge-route-both {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #fbbf24;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: inline-block;
}

/* Stat Counters */
.stat-box {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 14px 18px;
    text-align: center;
}
.stat-number {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
}
.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* Answer Card Styles */
.answer-box-baseline {
    background: rgba(30, 41, 59, 0.4);
    border-left: 4px solid #64748b;
    border-radius: 12px;
    padding: 18px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #cbd5e1;
}
.answer-box-hybrid {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(16, 185, 129, 0.05) 100%);
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 18px;
    font-size: 0.98rem;
    line-height: 1.6;
    color: #f1f5f9;
}

/* Status Indicator Dot */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
}

/* Custom buttons */
.stButton>button {
    border-radius: 12px;
    font-weight: 600;
    transition: all 0.2s;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# API / Standalone Helper Functions
# ---------------------------------------------------------
try:
    from app.ingest import ingest_pdf
    from app.agents import hybrid_app
    from app.baseline import baseline_answer
    from app.db import documents, tables_meta, table_rows, text_chunks, baseline_chunks
    from app.retrieval import table_catalog
    HAS_LOCAL_MODULES = True
except Exception:
    HAS_LOCAL_MODULES = False

def fetch_documents():
    try:
        r = requests.get(f"{API}/documents", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    if HAS_LOCAL_MODULES:
        try:
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
        except Exception:
            pass
    return []

def delete_doc(doc_id):
    try:
        requests.delete(f"{API}/documents/{doc_id}", timeout=3)
        return
    except Exception:
        pass
    if HAS_LOCAL_MODULES:
        for coll in (documents, tables_meta, table_rows, text_chunks, baseline_chunks):
            if coll == documents:
                coll.delete_one({"_id": doc_id})
            else:
                coll.delete_many({"doc_id": doc_id})

def fetch_tables_meta(doc_id):
    try:
        r = requests.get(f"{API}/documents/{doc_id}/tables", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    if HAS_LOCAL_MODULES:
        return table_catalog(doc_id)
    return []

def run_ingest(file_name, file_bytes):
    try:
        r = requests.post(f"{API}/ingest", timeout=900, files={"file": (file_name, file_bytes, "application/pdf")})
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    if HAS_LOCAL_MODULES:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return ingest_pdf(tmp_path, file_name)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    raise RuntimeError("Could not connect to API or local ingestion module.")

def run_baseline(doc_id, question):
    try:
        r = requests.post(f"{API}/ask/baseline", json={"doc_id": doc_id, "question": question}, timeout=300)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    if HAS_LOCAL_MODULES:
        return baseline_answer(question, doc_id)
    return {"answer": "Error: Unable to connect to backend", "chunks": []}

def run_hybrid(doc_id, question):
    try:
        r = requests.post(f"{API}/ask/hybrid", json={"doc_id": doc_id, "question": question}, timeout=300)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    if HAS_LOCAL_MODULES:
        s = hybrid_app.invoke({"question": question, "doc_id": doc_id})
        return {
            "answer": s["answer"],
            "route": s["route"],
            "route_reason": s.get("route_reason"),
            "table_result": s.get("table_result"),
            "text_result": s.get("text_result", [])
        }
    return {"answer": "Error: Unable to connect to backend", "route": "error", "route_reason": "Backend unavailable"}


# ---------------------------------------------------------
# Sidebar: Document Library & Multi-File Ingestion
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
        <div style="background: linear-gradient(135deg, #6366f1, #10b981); width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: bold; color: white;">⚡</div>
        <div>
            <h3 style="margin: 0; font-family: 'Outfit'; font-size: 1.2rem; font-weight: 700; color: #f8fafc;">Hybrid RAG</h3>
            <p style="margin: 0; font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.05em;">MONGODB ATLAS + GEMINI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="status-pill" style="margin-bottom: 18px;">
        <div class="pulse-dot"></div> Atlas Vector Search Ready
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📁 Upload & Ingest Documents")
    st.caption("Upload one or multiple PDF documents with tables and narrative prose.")

    uploaded_files = st.file_uploader(
        "Select PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        btn_label = f"🚀 Ingest {len(uploaded_files)} PDF{'s' if len(uploaded_files) > 1 else ''}"
        if st.button(btn_label, type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_files = len(uploaded_files)
            success_count = 0
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.markdown(f"⏳ **Ingesting:** `{uploaded_file.name}` ({idx+1}/{total_files})...")
                try:
                    res = run_ingest(uploaded_file.name, uploaded_file.getvalue())
                    st.toast(f"✅ Ingested {uploaded_file.name}", icon="📄")
                    st.session_state["doc"] = res
                    success_count += 1
                except Exception as err:
                    st.error(f"❌ Error ingesting `{uploaded_file.name}`:\n{err}")
                progress_bar.progress((idx + 1) / total_files)
            
            if success_count > 0:
                status_text.markdown(f"✨ **Ingested {success_count}/{total_files} file(s) successfully!**")
                time.sleep(1.5)
                st.rerun()

    st.markdown("---")
    st.markdown("### 📚 Knowledge Vault")

    docs = fetch_documents()
    if not docs:
        st.info("No documents ingested yet. Upload your PDF above to get started.")
        selected_doc_id = None
        current_doc = None
    else:
        doc_options = {d["doc_id"]: f"📄 {d['filename']}" for d in docs}
        selected_doc_id = st.selectbox(
            "Select Active Document to Query:",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x]
        )
        current_doc = next((d for d in docs if d["doc_id"] == selected_doc_id), None)

        if current_doc:
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; margin-top: 10px; font-size: 0.82rem;">
                <div style="color: #94a3b8;">Document ID: <code style="color: #cbd5e1;">{current_doc['doc_id']}</code></div>
                <div style="color: #94a3b8; margin-top: 4px;">Tables: <b style="color: #38bdf8;">{current_doc.get('tables', 0)}</b> | Rows: <b style="color: #34d399;">{current_doc.get('table_rows', 0)}</b> | Chunks: <b style="color: #a78bfa;">{current_doc.get('text_chunks', 0)}</b></div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🗑️ Delete This Document", use_container_width=True):
                delete_doc(selected_doc_id)
                st.toast("Document deleted", icon="🗑️")
                time.sleep(0.5)
                st.rerun()


# ---------------------------------------------------------
# Main Page Content
# ---------------------------------------------------------

# Hero Header
st.markdown("""
<div style="margin-top: 10px; margin-bottom: 24px;">
    <div class="hero-badge">⚡ Next-Gen Hybrid RAG Architecture</div>
    <div class="hero-title">One Database, Two Brains</div>
    <div class="hero-subtitle">
        High-precision semantic retrieval for narrative prose (<code style="color: #a5b4fc;">$vectorSearch</code>) combined with 
        LLM-generated aggregation pipelines (<code style="color: #34d399;">$match / $group / $sort</code>) over multi-page tables on MongoDB Atlas.
    </div>
</div>
""", unsafe_allow_html=True)

if not current_doc:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 50px 20px;">
        <div style="font-size: 3rem; margin-bottom: 12px;">📂</div>
        <h3 style="font-family: 'Outfit'; color: #f8fafc; margin-bottom: 8px;">No Document Selected</h3>
        <p style="color: #94a3b8; max-width: 500px; margin: 0 auto 20px auto;">
            Please upload and ingest your PDF in the left sidebar to unlock hybrid multi-agent querying.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Stats row for the active document
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color: #38bdf8;">{current_doc.get('tables', 0)}</div>
            <div class="stat-label">Extracted Tables</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color: #34d399;">{current_doc.get('table_rows', 0):,}</div>
            <div class="stat-label">Structured Rows</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color: #a78bfa;">{current_doc.get('text_chunks', 0)}</div>
            <div class="stat-label">Vector Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color: #fbbf24;">{current_doc.get('baseline_chunks', 0)}</div>
            <div class="stat-label">Baseline Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Example Question Chips
    st.markdown("#### 💡 Quick Test Prompts")
    q_cols = st.columns([1, 1, 1])
    with q_cols[0]:
        if st.button("📖 Text: Who won the 2010 Golden Ball?", use_container_width=True):
            st.session_state["user_q"] = "Who scored the winning goal in the 2010 World Cup final?"
    with q_cols[1]:
        if st.button("📊 Table: Brazil goals in 2002 World Cup?", use_container_width=True):
            st.session_state["user_q"] = "How many goals did Brazil score at the 2002 World Cup?"
    with q_cols[2]:
        if st.button("🔀 Multi-Hop: Golden Ball winner team goals?", use_container_width=True):
            st.session_state["user_q"] = "How far did the 2010 Golden Ball winner's team go, and how many goals did it score?"

    # Question Input
    default_q = st.session_state.get("user_q", "")
    question = st.text_input(
        "Ask any question about the document:",
        value=default_q,
        placeholder="e.g. Which players in Argentina's 2022 squad were born in 2000 or later?",
        key="main_query_input"
    )

    if question:
        query_body = {"doc_id": current_doc["doc_id"], "question": question}

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span style="font-size: 1.1rem; font-weight: 700; color: #94a3b8;">1. Conventional RAG</span>
                <span style="font-size: 0.75rem; background: rgba(148, 163, 184, 0.15); color: #cbd5e1; padding: 2px 8px; border-radius: 6px;">Baseline</span>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Executing conventional vector retrieval..."):
                try:
                    baseline_res = run_baseline(current_doc["doc_id"], question)
                except Exception as e:
                    baseline_res = {"answer": f"Error: {e}", "chunks": []}

            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #64748b;">
                <div class="answer-box-baseline">{baseline_res.get('answer', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Retrieved Flattened Chunks (Top 6)"):
                st.json(baseline_res.get("chunks", []))

        with col_right:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span style="font-size: 1.1rem; font-weight: 700; color: #6366f1;">2. Hybrid RAG (Two Brains)</span>
                <span style="font-size: 0.75rem; background: rgba(99, 102, 241, 0.2); color: #a5b4fc; padding: 2px 8px; border-radius: 6px;">Agentic Multi-Path</span>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Routing & executing LangGraph agents..."):
                try:
                    hybrid_res = run_hybrid(current_doc["doc_id"], question)
                except Exception as e:
                    hybrid_res = {"answer": f"Error: {e}", "route": "error", "route_reason": str(e)}

            route_type = hybrid_res.get("route", "text")
            badge_class = f"badge-route-{route_type}"

            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #6366f1;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <span class="{badge_class}">Route: {route_type.upper()}</span>
                    <span style="font-size: 0.8rem; color: #94a3b8; font-style: italic;">{hybrid_res.get('route_reason', '')}</span>
                </div>
                <div class="answer-box-hybrid">{hybrid_res.get('answer', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            if hybrid_res.get("table_result"):
                with st.expander("⚡ MongoDB Aggregation Pipeline & Table Query"):
                    st.markdown("**LLM Generated Aggregation Pipeline:**")
                    st.code(str(hybrid_res["table_result"].get("pipeline", [])), language="json")
                    st.markdown(f"**Rows returned:** {len(hybrid_res['table_result'].get('rows', []))}")
                    st.json(hybrid_res["table_result"].get("rows", []))

            if hybrid_res.get("text_result"):
                with st.expander("📄 Retrieved Narrative Text Chunks"):
                    st.json(hybrid_res.get("text_result", []))

    # Schema Catalog Inspector Tab
    st.markdown("---")
    with st.expander("📑 View Document Table Schema Catalog"):
        catalog_tables = fetch_tables_meta(current_doc["doc_id"])
        if catalog_tables:
            for tbl in catalog_tables:
                st.markdown(f"#### 🏷️ `{tbl.get('table_id')}`: {tbl.get('title')}")
                st.caption(f"Pages: {tbl.get('pages')} | Total Rows: {tbl.get('row_count')}")
                st.dataframe(tbl.get("columns", []), use_container_width=True)
        else:
            st.info("No structured tables cataloged for this document.")
