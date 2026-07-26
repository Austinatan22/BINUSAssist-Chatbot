FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# build-essential: PyStemmer (bm25s dependency) compiles a C extension at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Matches the GPU torch build verified against the host driver (CUDA 12.6). The wheel
# bundles its own CUDA runtime libs via pip deps, so no system CUDA toolkit is needed here.
# torchvision must come from the same cu126 index as torch — a CPU-built torchvision
# paired with a CUDA-built torch causes "operator torchvision::nms does not exist" on
# import (compiled-extension ABI mismatch), surfaced by transformers/docling at runtime.
# Kept in its own layer (~5GB download) so changes to requirements.txt below don't
# invalidate this cache and force a re-download.
RUN pip install --no-cache-dir torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY scripts ./scripts

# Seed the KB on first boot (idempotent -- see scripts/seed_if_empty.py) then start the API.
COPY backend/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]
