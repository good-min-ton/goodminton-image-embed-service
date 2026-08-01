def test_health_returns_status_and_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model_loaded": True,
        "rerank_loaded": True,
    }
