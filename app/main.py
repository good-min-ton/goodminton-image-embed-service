"""FastAPI entrypoint for the image-embedding microservice."""

import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SiglipImageProcessor,
    SiglipModel,
)

MODEL_NAME = "google/siglip-base-patch16-224"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_RERANK_DOCS = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_RERANK_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50 MP decompression-bomb guard (H4)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB raw-byte cap (H4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # H12: TORCH_NUM_THREADS should match the container's cpu allocation;
    # falls back to the host's logical cpu count outside a container.
    threads = int(os.environ.get("TORCH_NUM_THREADS", os.cpu_count() or 1))
    torch.set_num_threads(threads)

    app.state.model = SiglipModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(
        DEVICE
    )
    app.state.model.eval()
    app.state.processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)

    app.state.rerank_tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
    app.state.rerank_model = (
        AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, torch_dtype=_RERANK_DTYPE
        )
        .to(DEVICE)
        .eval()
    )

    yield

    app.state.model = None
    app.state.processor = None
    app.state.rerank_tok = None
    app.state.rerank_model = None


app = FastAPI(
    title="Goodminton Image Embed Service",
    version="0.1.0",
    description="SigLIP image-embedding microservice for visual product search.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "rerank_loaded": app.state.rerank_model is not None,
        "device": DEVICE,
    }


@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):  # noqa: B008
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="image exceeds max upload size")

    try:
        img = Image.open(io.BytesIO(data))
        # Pillow only *warns* (does not raise) between 1x and 2x MAX_IMAGE_PIXELS,
        # so enforce the pixel cap explicitly (H4) before decoding.
        if img.width * img.height > Image.MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=400, detail="image rejected by decompression-bomb guard"
            )
        img = img.convert("RGB")
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=400, detail="image rejected by decompression-bomb guard"
        )
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="invalid image data")

    inputs = app.state.processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features = app.state.model.get_image_features(**inputs)
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)
    return {"embedding": normalized.squeeze(0).float().cpu().tolist()}


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


@app.post("/rerank")
async def rerank(req: RerankRequest):
    if not req.query or not req.documents:
        raise HTTPException(status_code=400, detail="query and documents required")
    if len(req.documents) > MAX_RERANK_DOCS:
        raise HTTPException(status_code=400, detail="too many documents")
    pairs = [[req.query, doc] for doc in req.documents]
    inputs = app.state.rerank_tok(
        pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        scores = app.state.rerank_model(**inputs).logits.view(-1).float().cpu().tolist()
    return {"scores": scores}
