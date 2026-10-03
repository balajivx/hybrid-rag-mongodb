import os
from dotenv import load_dotenv

load_dotenv()

# Safely check Streamlit secrets if running inside Streamlit Cloud
_secrets = {}
try:
    import streamlit as _st
    _secrets = dict(_st.secrets)
except Exception:
    _secrets = {}

def get_setting(key, default=None):
    if key in os.environ and os.environ[key]:
        return os.environ[key]
    if key in _secrets and _secrets[key]:
        return str(_secrets[key])
    return default

MONGODB_URI = get_setting("MONGODB_URI", "")
DB_NAME = get_setting("DB_NAME", "hybrid_rag")
GEMINI_API_KEY = get_setting("GEMINI_API_KEY", "")
LLM_MODEL = get_setting("LLM_MODEL", "gemini-2.5-flash")
EMBED_MODEL = get_setting("EMBED_MODEL", "gemini-embedding-2")
EMBED_DIM = int(get_setting("EMBED_DIM", "768"))
VECTOR_INDEX = "chunk_vector_index"
MIN_TABLE_ROWS = 8  # smaller grids are treated as text
API_URL = get_setting("API_URL", "http://localhost:8000")
