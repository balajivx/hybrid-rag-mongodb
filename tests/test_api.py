from fastapi.testclient import TestClient
from app.api import api, AskRequest
from app.llm import get_client

client = TestClient(api)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ask_request_model():
    req = AskRequest(doc_id="test_doc", question="Who won?", api_key="custom-key-123")
    assert req.doc_id == "test_doc"
    assert req.question == "Who won?"
    assert req.api_key == "custom-key-123"

def test_get_client_dynamic_key():
    c = get_client("test-custom-api-key")
    assert c is not None
