"""What the service actually promises.

The existing tests prove the endpoint returns 768 normalised floats and rejects
malformed uploads. Both would still pass if the model were swapped for one that
embedded every image identically - the vectors would be the right shape and
carry no information. These tests cover the properties image search is built on:
the same product lands in the same place, different products do not, and the
transformations the browser applies on the way in do not move anything far.

Thresholds are deliberately loose against measured values (noted per test) so
they catch a broken or swapped model rather than normal drift.
"""

import math

from tests.fixtures import cosine, encode, racket_image, shirt_image


def embed(client, img, name="upload.png", content_type="image/png", fmt="PNG"):
    resp = client.post(
        "/embed/image",
        files={"file": (name, encode(img, fmt), content_type)},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["embedding"]


def test_the_same_bytes_always_embed_to_the_same_vector(client):
    """Search results must not shuffle between identical queries, and the
    backfill must not need re-running because yesterday's vectors drifted."""
    first = embed(client, racket_image())
    second = embed(client, racket_image())

    assert cosine(first, second) > 0.9999


def test_different_products_sit_further_apart_than_the_same_product(client):
    """The one property ranking depends on. If a racket is no closer to a
    smaller copy of itself than to a shirt, every search result is noise.

    Measured: 0.926 same, 0.653 different - a margin of 0.27. Asserted at 0.10
    so a genuinely degraded model fails and ordinary variation does not."""
    racket = embed(client, racket_image(512))
    same_product = embed(client, racket_image(128))
    other_product = embed(client, shirt_image(512))

    near = cosine(racket, same_product)
    far = cosine(racket, other_product)

    assert near - far > 0.10, f"same={near:.4f} different={far:.4f}"


def test_the_browsers_downscale_does_not_move_the_embedding(client):
    """lib/image-downscale.ts shrinks the long edge to 1024px and recompresses
    to JPEG before upload, so the vector indexed from a product photo and the
    vector from a customer's search of that same photo come from different
    bytes. They have to still match.

    Measured: 0.994 for this pair."""
    original = embed(client, racket_image(2048))
    as_the_browser_sends_it = embed(
        client, racket_image(1024), "upload.jpg", "image/jpeg", "JPEG"
    )

    assert cosine(original, as_the_browser_sends_it) > 0.90


def test_transparent_and_greyscale_uploads_are_accepted(client):
    """A PNG with an alpha channel and a greyscale photo are 4-channel and
    1-channel respectively. Handing either to the processor without converting
    to RGB raises inside the model, which would surface as a 500 on a perfectly
    ordinary upload."""
    for img in (racket_image().convert("RGBA"), racket_image().convert("L")):
        embedding = embed(client, img)

        assert len(embedding) == 768
        assert math.isclose(
            math.sqrt(sum(v * v for v in embedding)), 1.0, abs_tol=1e-4
        )
