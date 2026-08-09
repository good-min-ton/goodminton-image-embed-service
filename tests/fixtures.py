"""Drawn image fixtures.

Deliberately drawn rather than checked in as files: the suite stays free of
binary assets, and every property under test (size, format, colour mode) can be
varied by an argument instead of by adding another JPEG to the repository.

They are crude, but they are *structured* - an outline, a grid, a silhouette.
That matters. A solid colour swatch is not a useful fixture for this service:
SigLIP embeds a red square and a blue square 0.92 apart, because to the model
both are the same thing, "a flat colour field". Any test built on swatches
would pass whatever the model did.
"""

import io

from PIL import Image, ImageDraw


def racket_image(size: int = 512) -> Image.Image:
    """An oval head with a string grid and a handle."""
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    s = size / 512

    draw.ellipse([120 * s, 40 * s, 392 * s, 300 * s], outline=(20, 20, 20), width=int(10 * s))
    for x in range(140, 390, 18):
        draw.line([x * s, 45 * s, x * s, 295 * s], fill=(180, 180, 180), width=max(1, int(2 * s)))
    for y in range(50, 300, 18):
        draw.line([125 * s, y * s, 387 * s, y * s], fill=(180, 180, 180), width=max(1, int(2 * s)))
    draw.line([256 * s, 300 * s, 256 * s, 470 * s], fill=(30, 30, 30), width=int(18 * s))

    return img


def shirt_image(size: int = 512) -> Image.Image:
    """A short-sleeve silhouette - a different product category on the site."""
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    s = size / 512

    draw.polygon(
        [
            (160 * s, 90 * s),
            (352 * s, 90 * s),
            (420 * s, 160 * s),
            (370 * s, 215 * s),
            (352 * s, 180 * s),
            (352 * s, 430 * s),
            (160 * s, 430 * s),
            (160 * s, 180 * s),
            (142 * s, 215 * s),
            (92 * s, 160 * s),
        ],
        fill=(40, 90, 200),
        outline=(20, 20, 20),
    )

    return img


def encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def cosine(a: list[float], b: list[float]) -> float:
    """A plain dot product: /embed/image returns L2-normalised vectors, which is
    also how pgvector compares them on the shop-api side."""
    return sum(x * y for x, y in zip(a, b))
