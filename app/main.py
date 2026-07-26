"""FastAPI entrypoint for the image-embedding microservice."""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from transformers import SiglipImageProcessor, SiglipModel

MODEL_NAME = "google/siglip-base-patch16-224"


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
