import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag.ingestion import load_index
from backend.rag.models import init_models
from backend.rag.retrieval import build_fusion_retriever, build_reranker, retrieve_and_rerank

logging.basicConfig(level=logging.WARNING)

GOOD_QUERIES = [
    "What are the career prospects for Computer Science graduates?",
    "Bagaimana prospek karir bagi lulusan Ilmu Komputer?",
    "What are the learning outcomes of the Computer Science program?",
]

BAD_QUERIES = [
    "What is the capital of France?",
    "Write me a poem about the ocean",
    "How do I apply for academic leave?",
]


async def main() -> None:
    init_models()
    index = load_index()
    if index is None:
        print("No index found. Run scripts/seed_kb.py first.")
        return

    fusion_retriever = build_fusion_retriever(index)
    reranker = build_reranker()

    for label, queries in [("GOOD", GOOD_QUERIES), ("BAD", BAD_QUERIES)]:
        for q in queries:
            nodes = await retrieve_and_rerank(fusion_retriever, reranker, q)
            top_score = nodes[0].score if nodes else None
            print(f"[{label}] {q!r}")
            print(f"  top_score={top_score}")
            for n in nodes[:3]:
                print(
                    f"    score={n.score:.4f} retrieval_score={n.metadata.get('retrieval_score')} "
                    f"file={n.metadata.get('source_file')}"
                )


if __name__ == "__main__":
    asyncio.run(main())
