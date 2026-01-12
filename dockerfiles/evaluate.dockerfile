FROM ghcr.io/astral-sh/uv:python3.13-alpine AS base

WORKDIR /app

COPY uv.lock pyproject.toml ./
RUN uv sync --locked --no-install-project

COPY src/ src/
RUN uv sync --locked

# Instead of uv sync during build, use the following to speed up 
# iterative development with caching (not installing everything again)
#ENV UV_LINK_MODE=copy
#RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENTRYPOINT ["uv", "run", "src/xray_image_classifier/evaluate.py"]
