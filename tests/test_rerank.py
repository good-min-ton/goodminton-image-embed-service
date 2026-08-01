def test_rerank_returns_score_per_document(client):
    resp = client.post(
        "/rerank",
        json={
            "query": "giày cầu lông chống lật cổ chân",
            "documents": [
                "Giày cầu lông có công nghệ ổn định cổ chân chống lật.",
                "Vợt cầu lông khung carbon nhẹ.",
            ],
        },
    )
    assert resp.status_code == 200
    scores = resp.json()["scores"]
    assert len(scores) == 2 and all(isinstance(s, float) for s in scores)
    # the on-topic shoe doc must outscore the unrelated racket doc
    assert scores[0] > scores[1]


def test_rerank_rejects_empty_documents(client):
    resp = client.post("/rerank", json={"query": "x", "documents": []})
    assert resp.status_code == 400


def test_rerank_rejects_too_many_documents(client):
    resp = client.post("/rerank", json={"query": "x", "documents": ["d"] * 65})
    assert resp.status_code == 400


def test_health_reports_rerank_loaded(client):
    assert client.get("/health").json()["rerank_loaded"] is True
