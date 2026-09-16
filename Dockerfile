# ---- Frontend build stage --------------------------------------------
# Builds the React SPA (web/) into static files the Python stage below
# serves. This is the only UI -- the Streamlit `app.py` it replaced has
# been removed from the repo.
FROM node:20-slim AS frontend-builder
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# Same-origin deploy: the SPA and the FastAPI backend below are served by
# one process on one port, so built-in API calls must be relative, not
# hardcoded to the Vite dev server's localhost:8000 (see src/lib/api.ts).
ENV VITE_API_BASE_URL=""
RUN npm run build

# ---- Python app stage --------------------------------------------------
# CPU-only image: the judge runs on Amazon Bedrock, so nothing here needs
# a GPU or any local model weights.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# aiops-pipeline (pyproject.toml) is pulled from the private SNAIC-JP/Track2
# repo over git+https, so
# `uv sync` needs git installed plus a credential. The token is passed as a
# BuildKit secret (`docker build --secret id=track_repos_token,...`) and only
# ever touches a credential.helper that's configured, used, and unset inside
# a single RUN — never written to a config file that survives into a layer,
# so it can't be recovered from `docker history`/pulled image layers.
# git: uv installs aiops-pipeline from a git+https URL (see below).
# libgl1 / libglib2.0-0: opencv, pulled in by ultralytics for the stage-2
#   helmet detector, links against them. Without these `import ultralytics`
#   raises ImportError: libxcb.so.1 in this slim base image, which
#   helmet_detector.locate() catches -- so every helmet case silently fell
#   back to the judge's looser inline box instead of the detector's.
#   Verified against this exact image: these two are sufficient (libxcb
#   arrives as a dependency of libgl1).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first, separately from the app source, so this layer
# is cached across rebuilds that only change app code.
COPY pyproject.toml uv.lock ./
RUN --mount=type=secret,id=track_repos_token \
    git config --global credential.helper \
      '!f() { echo "username=x-access-token"; echo "password=$(cat /run/secrets/track_repos_token)"; }; f' \
    && uv sync --locked --no-install-project \
    && git config --global --unset-all credential.helper

COPY . .
RUN --mount=type=secret,id=track_repos_token \
    git config --global credential.helper \
      '!f() { echo "username=x-access-token"; echo "password=$(cat /run/secrets/track_repos_token)"; }; f' \
    && uv sync --locked \
    && git config --global --unset-all credential.helper

COPY --from=frontend-builder /app/web/dist ./web/dist

# Import the app once at build time. Without this, a syntax or import
# error only surfaces when the container starts -- which the deploy does
# AFTER removing the old container, so a typo takes the site down instead
# of failing the build while the running one is untouched. Imports only
# (boto3 clients are constructed lazily against no credentials here), so
# it talks to nothing.
RUN uv run python -c "import sys; sys.path.insert(0, 'src'); import ppe_pipeline_api.main"

EXPOSE 8501

# AWS_BEARER_TOKEN_BEDROCK / PPE_APP_PASSWORD / PPE_STORAGE_BACKEND etc.
# are read from the runtime environment (see .env.example) — never baked
# into the image. Runs the FastAPI + React stack: main.py serves both the
# API and the built SPA in web/dist on the same port.
ENTRYPOINT ["uv", "run", "uvicorn", "ppe_pipeline_api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8501"]
