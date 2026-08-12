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
# Bake by calling the SAME function startup calls, not by listing the models
# again here. The list has now drifted twice: first the reranker was missing,
# then SigLIP's text tokenizer, and both times the image shipped fine and the
# container died on the server instead - HF_HUB_OFFLINE below turns a file
# missing from the cache into "expected str, bytes or os.PathLike object, not
# NoneType", which names neither the model nor the file.
#
# The previous attempt imported the model names from app.main and claimed that
# settled it. It did not: the names never drifted, the per-model component list
# did. One shared function is the only version of this that cannot drift.
RUN uv run python -c "from app.main import load_models; load_models()"

# H11: guarantee no runtime Hub connectivity — serve the baked model from cache only.
# (Also keeps startup fast: no Hub connectivity check to time out on.)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
