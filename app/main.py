"""FastAPI entrypoint for the image-embedding microservice."""

from fastapi import FastAPI

app = FastAPI(
    title="Goodminton Image Embed Service",
    version="0.1.0",
    description="SigLIP image-embedding microservice for visual product search.",
)
