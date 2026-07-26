import io
import math

from PIL import Image


def _solid_color_png_bytes(color=(200, 30, 30), size=(64, 64)) -> bytes:
    """Tiny in-memory PNG fixture — no network dependency."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_embed_image_returns_768_dim_normalized_vector(client):
    png_bytes = _solid_color_png_bytes()
    resp = client.post(
        "/embed/image",
        files={"file": ("swatch.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    embedding = resp.json()["embedding"]
    assert len(embedding) == 768
    norm = math.sqrt(sum(v * v for v in embedding))
    assert math.isclose(norm, 1.0, abs_tol=1e-4)
