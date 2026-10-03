from fastapi.testclient import TestClient
from app.api import api

client = TestClient(api)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
