import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.environ["MONGODB_URI"]
DB_NAME = os.getenv("DB_NAME", "hybrid_rag")
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
VECTOR_INDEX = "chunk_vector_index"
MIN_TABLE_ROWS = 8  # smaller grids are treated as text
API_URL = os.getenv("API_URL", "http://localhost:8000")
