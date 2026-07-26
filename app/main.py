"""FastAPI entrypoint for the image-embedding microservice."""

import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from transformers import SiglipImageProcessor, SiglipModel

MODEL_NAME = "google/siglip-base-patch16-224"

Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50 MP decompression-bomb guard (H4)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB raw-byte cap (H4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # H12: TORCH_NUM_THREADS should match the container's cpu allocation;
    # falls back to the host's logical cpu count outside a container.
    threads = int(os.environ.get("TORCH_NUM_THREADS", os.cpu_count() or 1))
    torch.set_num_threads(threads)

    app.state.model = SiglipModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    app.state.model.eval()
    app.state.processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)

    yield

    app.state.model = None
    app.state.processor = None


app = FastAPI(
    title="Goodminton Image Embed Service",
    version="0.1.0",
    description="SigLIP image-embedding microservice for visual product search.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
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

    inputs = app.state.processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = app.state.model.get_image_features(**inputs)
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)
    return {"embedding": normalized.squeeze(0).tolist()}
