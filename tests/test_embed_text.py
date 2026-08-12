import math


def test_embed_text_returns_768_normalized(client):
    resp = client.post("/embed/text", json={"text": "vợt cầu lông màu đỏ cán ngắn"})
    assert resp.status_code == 200
    emb = resp.json()["embedding"]
    assert len(emb) == 768
    norm = math.sqrt(sum(x * x for x in emb))
    assert abs(norm - 1.0) < 1e-3  # L2-normalized like the image path


def test_embed_text_handles_long_query_over_64_tokens(client):
    # SigLIP max_position_embeddings=64; a long VN description tokenizes to >64
    # tokens. Without truncation=True get_text_features raises ValueError (508>64).
    long = "vợt cầu lông màu đỏ cán ngắn nhẹ cho người mới chơi tập luyện hàng ngày " * 6
    resp = client.post("/embed/text", json={"text": long})
    assert resp.status_code == 200
    assert len(resp.json()["embedding"]) == 768


def test_embed_text_rejects_empty(client):
    assert client.post("/embed/text", json={"text": "   "}).status_code == 400


def test_health_reports_text_tokenizer_loaded(client):
    assert client.get("/health").json()["text_tokenizer_loaded"] is True
