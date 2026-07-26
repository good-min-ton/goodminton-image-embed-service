import io

from PIL import Image

from app import main as main_module


def _solid_color_png_bytes(size=(64, 64), color=(10, 200, 10)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_embed_image_rejects_non_image_content_type(client):
    resp = client.post(
        "/embed/image",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 400


def test_embed_image_rejects_decompression_bomb(client, monkeypatch):
    # 40x40 = 1_600px vs MAX 1_000 -> 1.6x, i.e. INSIDE Pillow's 1x-2x
    # "warn-only" band where it does NOT raise DecompressionBombError.
    # This locks the explicit pixel-cap check (H4) — a catch-only guard fails here.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000)
    png_bytes = _solid_color_png_bytes(size=(40, 40))
    resp = client.post(
        "/embed/image",
        files={"file": ("big.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 400


def test_embed_image_rejects_oversize_upload(client, monkeypatch):
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 10)
    png_bytes = _solid_color_png_bytes()
    resp = client.post(
        "/embed/image",
        files={"file": ("swatch.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 400


def test_no_batch_embed_endpoint_exists(client):
    """H6: no URL-fetch / batch endpoint — only /embed/image + /health exist."""
    resp = client.post(
        "/embed/images",
        files={"file": ("x.png", b"", "image/png")},
    )
    assert resp.status_code == 404
