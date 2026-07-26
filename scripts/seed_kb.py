import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.rag.ingestion import build_index, load_documents
from backend.rag.models import init_models

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    init_models()

    nodes = load_documents(settings.documents_dir)
    if not nodes:
        print(f"No PDF/DOCX files found in {settings.documents_dir}. Add documents and re-run.")
        return

    build_index(nodes)

    counts: dict[str, int] = {}
    for node in nodes:
        source = node.metadata["source_file"]
        counts[source] = counts.get(source, 0) + 1

    print("\nIngestion summary:")
    for fname, count in counts.items():
        print(f"  {fname}: {count} chunks")
    print(f"Total chunks indexed: {len(nodes)}")
    print(f"Collection: {settings.chroma_collection_name}")
    print(f"Persisted to: {settings.chroma_persist_dir}")


if __name__ == "__main__":
    main()
