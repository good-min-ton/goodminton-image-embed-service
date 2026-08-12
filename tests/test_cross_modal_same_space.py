"""Guards the load-bearing "same joint space" guarantee behind text->image search.

If tokenization/dtype/normalization ever regresses in a way that still returns a
unit 768-vector (Task 1's image-embed guard would stay green) but breaks
comparability between the text and image towers, this test catches it: cosine
between a matching text description and an image must beat cosine against an
unrelated one.

Uses English labels deliberately -- SigLIP is English-trained, so this keeps
the test a same-space directionality check, not a Vietnamese-quality check.
"""


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))  # both already L2-normalized


def _img_embed(client, path):
    with open(path, "rb") as f:
        r = client.post("/embed/image", files={"file": ("racket.jpg", f.read(), "image/jpeg")})
    assert r.status_code == 200
    return r.json()["embedding"]


def _txt_embed(client, text):
    r = client.post("/embed/text", json={"text": text})
    assert r.status_code == 200
    return r.json()["embedding"]


def test_text_and_image_share_joint_space(client):
    # A racket+shuttlecock photo must be cosine-closer to a matching EN
    # description than to an unrelated one.
    img = _img_embed(client, "tests/fixtures/racket.jpg")
    match = _txt_embed(client, "a badminton racket")
    mismatch = _txt_embed(client, "a pair of running shoes")

    assert len(img) == len(match) == len(mismatch) == 768

    cos_match = _cos(img, match)
    cos_mismatch = _cos(img, mismatch)

    assert cos_match > cos_mismatch  # same-space directionality
