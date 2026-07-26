#!/bin/sh
# Container entrypoint: seed the knowledge base if it isn't already built (idempotent -- a
# persisted vectorstore volume is reused, so this is a no-op on restart), then start the API.
# See scripts/seed_if_empty.py for what "seed" covers (documents + scraped URLs + faculty).
set -e

python scripts/seed_if_empty.py

exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
