import json
import time
import os
import re
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, LLM_MODEL, EMBED_MODEL, EMBED_DIM

FALLBACK_LLM_MODELS = [
    LLM_MODEL,
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
]

def get_client(api_key=None):
    """Dynamically get or create a Google GenAI client with active API key."""
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not key:
        raise ValueError("Gemini API Key is missing. Please provide a valid Google Gemini API Key in the UI or configuration.")
    return genai.Client(api_key=key)

def _extract_retry_delay(exc, default_delay=5.0):
    """Extract recommended retry delay from API error message or details."""
    err_str = str(exc)
    delay = default_delay
    match = re.search(r"retry in ([\d\.]+)s", err_str)
    if match:
        try:
            delay = float(match.group(1)) + 1.0
        except Exception:
            delay = default_delay
    else:
        match = re.search(r"'retryDelay': '(\d+)s'", err_str)
        if match:
            try:
                delay = float(match.group(1)) + 1.0
            except Exception:
                delay = default_delay
    return min(max(delay, 2.0), 45.0)

def embed(texts, task="RETRIEVAL_DOCUMENT", batch=50, max_retries=5, api_key=None):
    """Embed a list of strings with dynamic API key and automatic rate-limit backoff."""
    if not texts:
        return []
    client = get_client(api_key)
    vectors = []
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i + batch]
        for attempt in range(max_retries):
            try:
                res = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch_texts,
                    config=types.EmbedContentConfig(
                        task_type=task,
                        output_dimensionality=EMBED_DIM
                    ),
                )
                vectors.extend(e.values for e in res.embeddings)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                delay = _extract_retry_delay(e, default_delay=min(2 ** attempt * 2, 30))
                print(f"[Gemini Embed] Rate limit hit. Retrying batch in {delay:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
    return vectors

def generate(prompt, json_mode=False, max_retries=3, api_key=None):
    """Generate LLM content with dynamic API key, multi-model fallback, and rate-limit backoff."""
    client = get_client(api_key)
    cfg = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json" if json_mode else "text/plain",
    )
    last_error = None
    for model_name in FALLBACK_LLM_MODELS:
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=cfg
                )
                return json.loads(resp.text) if json_mode else resp.text
            except Exception as e:
                last_error = e
                err_msg = str(e)
                # If quota is exhausted on this specific model, immediately try next fallback model
                if "GenerateRequestsPerDay" in err_msg or "RESOURCE_EXHAUSTED" in err_msg and "limit: 20" in err_msg:
                    print(f"[Gemini Generate] Model {model_name} quota exceeded. Falling back to alternative model...")
                    break
                if attempt == max_retries - 1:
                    break
                delay = _extract_retry_delay(e, default_delay=min(2 ** attempt * 2, 20))
                print(f"[Gemini Generate] Model {model_name} busy. Retrying in {delay:.1f}s...")
                time.sleep(delay)
    if last_error:
        raise last_error
