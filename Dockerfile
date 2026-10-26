# ---- Frontend build stage --------------------------------------------
# Builds the React SPA (web/) into static files the Python stage below
# serves. This is the only UI -- the Streamlit `app.py` it replaced has
# been removed from the repo.
# Both base images are mirrored into this account's own ECR by the
# `track2-app-base` repo, so a build pulls them in-region rather than from
# Docker Hub or GitHub. That is the whole difference between a four minute
# deploy and an eighteen minute one:
#
#   17 Sep   12 builds   3.9 - 6.9 min   (layers happened to be on the host)
#   24 Sep   same code   4.0 - 18.3 min  (bimodal: cached host vs cold pull)
#
# Refresh them with the mirror buildspec when a newer base is wanted --
# terraform/app/base_images.tf explains why that is deliberate.
FROM 125192671932.dkr.ecr.ap-southeast-1.amazonaws.com/track2-app-base:node-20-slim AS frontend-builder
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
# CPU-only image. torch and torchvision come from the CPU wheel index
# (see [tool.uv.sources] in pyproject.toml); nothing here needs a GPU.
# Training runs on SageMaker, in a different image entirely
# (model_training/training-container).
FROM 125192671932.dkr.ecr.ap-southeast-1.amazonaws.com/track2-app-base:uv-python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# libgl1 / libglib2.0-0: opencv, pulled in by ultralytics, links against
#   them. Without these `import ultralytics` raises ImportError on
#   libxcb.so.1 in this slim base image. ultralytics is imported for real
#   here -- registering a checkpoint slices its detection head down to the
#   chosen classes -- so this is not optional. Verified against this exact
#   image: these two are sufficient (libxcb arrives with libgl1).
#
# git is deliberately NOT installed any more. It was here solely because
# `uv sync` resolved aiops-pipeline from a private git+https URL, which in
# turn meant threading a GitHub token through the build as a BuildKit
# secret and unsetting the credential helper inside each RUN so it could
# not survive into a layer. That dependency went with the PPE judge, so
# every package now comes from an index and the build needs no credential
# at all. The buildspec still writes /tmp/token; it is simply unused, and
# an unused BuildKit secret is not an error.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first, separately from the app source, so this layer
# is cached across rebuilds that only change app code.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY . .
RUN uv sync --locked

COPY --from=frontend-builder /app/web/dist ./web/dist

# Import the app once at build time. Without this, a syntax or import
# error only surfaces when the container starts -- which the deploy does
# AFTER removing the old container, so a typo takes the site down instead
# of failing the build while the running one is untouched. Imports only
# (boto3 clients are constructed lazily against no credentials here), so
# it talks to nothing.
RUN uv run python -c "import sys; sys.path.insert(0, 'src'); import ppe_pipeline_api.main"

EXPOSE 8501

# PPE_APP_PASSWORD and the table/bucket/pipeline names are read from the
# runtime environment (see .env.example) — never baked into the image. Runs the FastAPI + React stack: main.py serves both the
# API and the built SPA in web/dist on the same port.
ENTRYPOINT ["uv", "run", "uvicorn", "ppe_pipeline_api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8501"]
