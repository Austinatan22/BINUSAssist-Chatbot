"""Seed the knowledge base ONLY if it isn't already built -- the container-startup guard so
`docker compose up` on a fresh deploy answers out of the box instead of booting with an empty
index (every query falling back). Idempotent: a persisted vectorstore (mounted volume) is left
untouched, so restarts and redeploys don't re-seed or re-crawl.

This does the FULL seed, matching /admin/reindex -- not just scripts/seed_kb.py (which only
covers local documents_dir files). The complete KB is:
  - local program documents (documents_dir, PDF/DOCX)
  - every previously-scraped URL (scraped_urls.json), re-fetched with the last-known-good
    cache fallback (a URL that can't be fetched contributes its cached chunks, never nothing)
  - faculty (rebuilt from faculty_snapshot.json offline; a first-ever run with no snapshot
    bootstraps one live crawl -- see _faculty_roster_nodes)

Exit code is always 0 for an already-seeded or empty-source case: a fresh deploy with no
documents yet should still start the server (it just falls back until content is added via the
admin panel), so this must never block container startup.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.rag.ingestion import (
    build_index,
    load_documents,
    load_index,
    load_scraped_urls,
    scrape_url_cached,
)
from backend.rag.models import init_models

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seed_if_empty")


def main() -> None:
    init_models()

    if load_index() is not None:
        logger.info("Knowledge base already present -- skipping seed (persisted volume reused).")
        return

    logger.info("No index found -- seeding the knowledge base from scratch...")

    nodes = load_documents(settings.documents_dir)
    logger.info("  documents: %d chunks from %s", len(nodes), settings.documents_dir)

    urls = load_scraped_urls()
    if urls:
        logger.info("  re-fetching %d scraped URL(s) (with last-known-good fallback)...", len(urls))
        failed = []
        for url in urls:
            url_nodes, from_cache = scrape_url_cached(url)
            if url_nodes:
                nodes.extend(url_nodes)
                if from_cache:
                    logger.info("    %s -> served from cache (stale)", url)
            else:
                failed.append(url)
                logger.warning("    %s -> could not fetch and no cached content", url)
        if failed:
            logger.warning("  %d URL(s) had no content this run: %s", len(failed), failed)

    if not nodes:
        logger.warning(
            "No documents or scraped content found -- starting with an EMPTY knowledge base. "
            "Add program documents to %s (or via the admin panel) and re-seed.",
            settings.documents_dir,
        )
        return

    build_index(nodes)
    logger.info("Seed complete: %d chunks indexed, persisted to %s", len(nodes), settings.chroma_persist_dir)


if __name__ == "__main__":
    main()
