# Image-embedding microservice — SigLIP (see design spec §1 embed-service).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Cache deps layer riêng
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy code
COPY . .
RUN uv sync --frozen --no-dev

# H11: bake the SigLIP model into the image at build time so the container
# never needs internet access at runtime. This means `docker build` needs
# internet once; for fully offline environments, build here and transfer
# the image with `docker save` / `docker load`.
RUN uv run python -c "\
from transformers import SiglipImageProcessor, SiglipModel; \
SiglipModel.from_pretrained('google/siglip-base-patch16-224', use_safetensors=True); \
SiglipImageProcessor.from_pretrained('google/siglip-base-patch16-224')"

# H11: guarantee no runtime Hub connectivity — serve the baked model from cache only.
# (Also keeps startup fast: no Hub connectivity check to time out on.)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
