# Image-embedding microservice — SigLIP (see design spec §1 embed-service).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Cache deps layer riêng
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy code
COPY . .
RUN uv sync --frozen --no-dev

# H11: bake every model the app loads into the image at build time so the
# container never needs internet access at runtime. This means `docker build`
# needs internet once; for fully offline environments, build here and transfer
# the image with `docker save` / `docker load`.
#
# The names come from app.main rather than being repeated here. They used to be
# written out twice, and when the reranker was added to the app nobody added it
# here - so the image shipped with only SigLIP cached while startup asked for a
# second model, and HF_HUB_OFFLINE below turned that into a crash loop. Importing
# the constants makes it impossible for the two lists to drift apart again.
RUN uv run python -c "\
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, \
                          SiglipImageProcessor, SiglipModel); \
from app.main import MODEL_NAME, RERANK_MODEL; \
SiglipModel.from_pretrained(MODEL_NAME, use_safetensors=True); \
SiglipImageProcessor.from_pretrained(MODEL_NAME); \
AutoTokenizer.from_pretrained(RERANK_MODEL); \
AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)"

# H11: guarantee no runtime Hub connectivity — serve the baked model from cache only.
# (Also keeps startup fast: no Hub connectivity check to time out on.)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
