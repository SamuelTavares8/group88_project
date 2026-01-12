FROM ghcr.io/astral-sh/uv:python3.13-alpine AS base

# copy lock + project metadata and install pinned deps
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN uv sync --frozen --no-install-project

# copy source and install project (dev deps not installed)
COPY src src/
RUN uv sync --frozen

# Instead of uv sync during build, use the following to speed up 
# iterative development with caching (not installing everything again)
#ENV UV_LINK_MODE=copy
#RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENTRYPOINT ["uv", "run", "src/xray_image_classifier/train.py"]
