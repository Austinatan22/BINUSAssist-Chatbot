import asyncio
import re
from pathlib import Path

import torch
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

from backend.config import settings


def build_fusion_retriever(index: VectorStoreIndex) -> QueryFusionRetriever:
    """Combine dense vector search and BM25 sparse search via reciprocal rank fusion.

    num_queries=1 disables QueryFusionRetriever's default LLM-based query expansion
    (which would otherwise fire an extra LLM call per request).
    """
    dense_retriever = index.as_retriever(similarity_top_k=settings.retrieval_top_k)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=settings.retrieval_top_k
    )
    return QueryFusionRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        similarity_top_k=settings.fusion_top_k,
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=True,
    )


def build_reranker() -> SentenceTransformerRerank:
    reranker = SentenceTransformerRerank(
        model=settings.reranker_model_name,
        top_n=settings.rerank_top_n,
        device=settings.reranker_device,
        keep_retrieval_score=True,
    )
    if settings.reranker_device == "cuda":
        # fp32 weights for this 568M-param cross-encoder used ~4-4.5GB of VRAM; halving
        # precision cuts that roughly in half without a meaningful accuracy loss for reranking.
        # In-place .half() only -- reassigning the .model attribute triggers a CrossEncoder
        # property-setter side effect that breaks attention_mask handling (confirmed via repro).
        reranker._model.model.half()
        # .half() frees the fp32 weights but torch keeps their blocks in its caching allocator,
        # so the process goes on RESERVING the fp32 load spike for its whole lifetime even though
        # it is using half of it. Measured on a 10GB card, with the embedder loaded too: reserved
        # while serving 3770MB -> 3172MB, so 598MB goes back to the driver. Identical allocated
        # memory (2174MB either way) and bit-identical rerank scores, since nothing about the
        # model changes -- this only hands back cache.
        #
        # It does NOT lower the peak (3737MB): fp32 is still materialized during the load. Doing
        # better would mean constructing the CrossEncoder in fp16 up front, and there is no way to
        # reach that from here -- SentenceTransformerRerank takes no model_kwargs and CrossEncoder's
        # dtype lives behind them. Building the fp16 model separately and swapping it in is worse,
        # not better: SentenceTransformerRerank still loads its own fp32 copy in __init__, so the
        # model ends up in memory twice (measured 3257MB allocated vs 1091MB for the approach
        # above). At 3.7GB peak on a 10GB card the spike is not worth contorting this for; holding
        # it forever was the part worth fixing.
        torch.cuda.empty_cache()
    return reranker


async def retrieve_and_rerank(
    fusion_retriever: QueryFusionRetriever,
    reranker: SentenceTransformerRerank,
    query: str,
    extra_queries: list[str] | None = None,
) -> list[NodeWithScore]:
    """Fused dense+BM25 retrieval over the whole KB, reranked and capped to
    settings.rerank_top_n. The single entry point for un-scoped retrieval -- pass
    `extra_queries` (rewritten paraphrases) on the low-confidence retry path (R-08) and
    the candidate pools are merged and each node is credited with its BEST rerank score
    across every query tried, not just the original phrasing.

    Why best-across-queries and not just the original: an earlier version reranked the
    merged retry pool against only the original query, which discarded matches a rewrite
    legitimately found -- confirmed live, "what are the steps to apply as an undergraduate
    student?" rewritten to "undergraduate program admission requirements" retrieved the
    right page and reranked at 0.166 against its own phrasing, but only 0.006 against the
    filler-stripped original, so it fell back despite the right content sitting in the
    pool. rewrite_query only ever paraphrases the same question, so crediting a chunk's
    best score across phrasings doesn't relax the bar, it just stops penalizing a genuine
    match for scoring badly against a worse phrasing.

    All queries are reranked in ONE batched cross-encoder call (see _max_score_rerank),
    not one per query. For the common single-query case (extra_queries is None) this is
    just a fused retrieve + a one-query rerank; the merge/dedup collapses to a no-op.
    aretrieve(), not retrieve(): QueryFusionRetriever's sync retrieve() spins up a nested
    event loop that breaks under FastAPI's running loop.
    """
    all_queries = [query] + list(extra_queries or [])
    results = await asyncio.gather(*(fusion_retriever.aretrieve(q) for q in all_queries))

    seen: set[str] = set()
    merged: list[NodeWithScore] = []
    for nodes in results:
        for node in nodes:
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            merged.append(node)
    if not merged:
        return []

    # Reranking is CPU/GPU-bound; offload so it doesn't block the event loop for other
    # concurrent requests.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _max_score_rerank, reranker, merged, all_queries)
    merged.sort(key=lambda n: n.score if n.score is not None else float("-inf"), reverse=True)
    return merged[: settings.rerank_top_n]


_YEAR_SUFFIX_RE = re.compile(r"_\d{4}$")


def get_program_catalog(index: VectorStoreIndex) -> dict[str, str]:
    """{display_name: source_file} for every locally-ingested document (PDF/DOCX/XLSX/CSV)
    currently in the index -- excludes URL-scraped sources, which aren't standalone
    "programs" to compare against each other. Used to let an LLM classifier pick from
    real, currently-indexed programs only (see detect_named_programs), so it can't
    hallucinate a match against something not actually in the KB. Reads the
    in-memory docstore rather than round-tripping to Chroma, since this runs on every
    chat request.
    """
    source_files = {
        node.metadata.get("source_file")
        for node in index.docstore.docs.values()
        if node.metadata.get("source_file")
        and not node.metadata["source_file"].startswith("http")
    }
    catalog: dict[str, str] = {}
    for source_file in source_files:
        stem = _YEAR_SUFFIX_RE.sub("", Path(source_file).stem)
        display_name = re.sub(r"_+", " ", stem).strip()
        catalog[display_name] = source_file
    return catalog


async def retrieve_for_named_programs(
    index: VectorStoreIndex,
    reranker: SentenceTransformerRerank,
    query: str,
    source_files: list[str],
    per_program_top_n: int = 4,
    balanced: bool = False,
    extra_queries: list[str] | None = None,
    max_nodes: int | None = None,
) -> list[NodeWithScore]:
    """Retrieves the best chunks for `query` from EACH of source_files independently --
    metadata-filtered dense retrieval per document, then reranked and merged. Used for
    both comparison mode (2-3 source_files) and single-program-scoped retrieval (exactly
    1) -- in both cases the point is restricting candidates to only the named program(s)'
    own document(s). Also used for the supplementary-sources retry (backend/main.py),
    where source_files can include every scraped URL at once (dozens), not just 1-3
    programs.

    `extra_queries` (rewritten paraphrases, R-08 -- see retrieve_and_rerank) mirrors that
    function's multi-query support: each source is retrieved against every query and the
    per-source candidate pools merged/deduped before a single batched rerank credits each
    node with its best score across all queries tried. Found live: a program-scoped
    Indonesian-language question ("Apa saja capaian pembelajaran program studi Ilmu
    Komputer?") scored 0.016 against Computer Science's own catalog PDF (written entirely
    in English/ACM terminology, so cross-lingual dense retrieval alone can't bridge the
    gap) even though a same-topic ENGLISH question scored 0.997 against the identical
    document -- a pure vocabulary-mismatch problem, exactly what rewrite_query exists to
    fix, but it was only ever wired into the unscoped open-retrieval path before this.

    A comparison question run through the normal fused top-k can be dominated by
    whichever program's chunks score marginally higher overall, starving the other side
    of any representation in the final context even though it has perfectly good
    matching content of its own. For the single-program case, an unrestricted top-k
    risks a different failure: boilerplate content embedded in a DIFFERENT program's
    document (a Free Electives cross-listing table, a stray "Minor Program: X" section)
    competing for this program's answer just because it lexically overlaps. Filtering to
    one document at a time first prevents both -- every named program gets a fair,
    independently-ranked slice of the answer, and nothing from an unnamed program's
    document is ever a candidate at all.
    Dense-only (no BM25 leg): this path is scoped to one document at a time, and the
    per-program queries submitted here are usually short ("{program} curriculum"), which
    is exactly BM25's weak spot -- dense cosine similarity carries this fine on its own.

    Retrieval is parallelized (asyncio.gather) and reranking is ONE batched cross-encoder
    call across every source's candidates (see _max_score_rerank), not one call per
    source -- found live, once source_files could hold 40+ supplementary URLs at once,
    the original one-rerank-call-per-source loop pushed a single request's retrieval
    stage past 20 seconds (46 sequential small GPU calls instead of one larger one, the
    same shape of regression _max_score_rerank was already written to fix for the
    paraphrase-retry path). Grouping the batched results back by source before taking
    each one's own top-N (rather than one global top-N) is what preserves the fairness
    guarantee above -- a single global cut would let one source's many good chunks crowd
    out the others entirely.

    `max_nodes` overrides the final cap (default settings.rerank_top_n, same budget the
    normal fused-retrieval path enforces). Only ever raised by a caller that KNOWS its
    candidate chunks are unusually small -- e.g. chat_service.py's campus-balanced
    tuition retry, where each chunk is one program/campus/year row (~65 tokens, see
    ingestion.py's _tuition_fee_row_nodes) rather than a generic multi-paragraph chunk,
    so covering every campus costs less total context than the default cap used to
    spend on far fewer, far noisier chunks.
    """
    all_queries = [query] + list(extra_queries or [])
    # settings.retrieval_top_k (20) is per SOURCE here, so a wide fan-out multiplies it and
    # reranking dominates this path (measured: 2.02s for 260 real chunks, ~7.8ms each,
    # against 1.47s for the 48 dense retrievals that produced them). Lowering it is
    # tempting and was tried: at 6 per source, "Di kampus mana saja ada Computer Science?"
    # and "Is there an entrance exam for Computer Science?" both stopped answering. Dense
    # top-k is the RECALL stage and the cross-encoder is the precision stage -- the whole
    # value of reranking is promoting a chunk that dense ranked low, so starving it of
    # candidates defeats it. Cut the number of SOURCES instead (see chat_service's
    # _ASPECT_URL_FAMILIES), which shrinks the same pool without touching per-source recall.

    async def _retrieve_one(source_file: str) -> tuple[str, list[NodeWithScore]]:
        retriever = index.as_retriever(
            similarity_top_k=settings.retrieval_top_k,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="source_file", value=source_file)]
            ),
        )
        per_query_results = await asyncio.gather(*(retriever.aretrieve(q) for q in all_queries))
        seen: set[str] = set()
        merged: list[NodeWithScore] = []
        for nodes in per_query_results:
            for node in nodes:
                if node.node_id in seen:
                    continue
                seen.add(node.node_id)
                merged.append(node)
        return source_file, merged

    results = await asyncio.gather(*(_retrieve_one(sf) for sf in source_files))

    all_candidates: list[NodeWithScore] = []
    source_of: dict[str, str] = {}
    for source_file, nodes in results:
        for node in nodes:
            all_candidates.append(node)
            source_of[node.node_id] = source_file

    if not all_candidates:
        return []

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _max_score_rerank, reranker, all_candidates, all_queries)

    by_source: dict[str, list[NodeWithScore]] = {}
    for node in all_candidates:
        by_source.setdefault(source_of[node.node_id], []).append(node)
    for nodes in by_source.values():
        nodes.sort(key=lambda n: n.score if n.score is not None else float("-inf"), reverse=True)

    cap = max_nodes if max_nodes is not None else settings.rerank_top_n

    if balanced:
        # Comparison mode: fill the cap round-robin across programs (each program's #1,
        # then each program's #2, ...) so neither side is starved of context. Without
        # this, the plain global top-N below lets whichever program reranks higher
        # overall take nearly all the slots -- confirmed live, a 2-program comparison came
        # back 4 chunks from one program and only 1 from the other, so the answer couldn't
        # actually compare them. per_program_top_n still bounds any one program's share.
        selected: list[NodeWithScore] = []
        for rank in range(per_program_top_n):
            for source_file in source_files:
                nodes = by_source.get(source_file, [])
                if rank < len(nodes):
                    selected.append(nodes[rank])
            if len(selected) >= cap:
                break
        selected = selected[:cap]
        # Round-robin above decides WHICH nodes are kept (the balance guarantee); sorting
        # the kept set by score only orders them, so selected[0] is the true best score
        # (the confidence-gate backstop the caller checks) and the LLM sees highest-
        # relevance context first, without disturbing the per-program balance.
        selected.sort(key=lambda n: n.score if n.score is not None else float("-inf"), reverse=True)
        return selected

    all_nodes = []
    for source_file in source_files:
        all_nodes.extend(by_source.get(source_file, [])[:per_program_top_n])
    # Sorted globally (not just per-program) so all_nodes[0] is a meaningful "best score
    # across everything retrieved" -- callers use it as a confidence-gate backstop, since
    # a program-name match alone (detect_named_programs) isn't proof the retrieved
    # content is actually relevant if that classification was somehow wrong.
    all_nodes.sort(key=lambda n: n.score if n.score is not None else float("-inf"), reverse=True)
    # Capped overall (default settings.rerank_top_n, same final budget the normal
    # fused-retrieval path enforces via build_reranker's top_n) -- len(source_files) *
    # per_program_top_n has no upper bound on its own, and an uncapped node list here
    # means an uncapped prompt size. Found live: pushed a single request's prompt past
    # the LLM provider's per-minute token limit (413 Payload Too Large) even though the daily
    # budget (#3.2) had plenty of room left -- a different limit than the one that guards against.
    return all_nodes[:cap]


def _max_score_rerank(
    reranker: SentenceTransformerRerank, nodes: list[NodeWithScore], queries: list[str]
) -> None:
    """Scores every node against every query in ONE batched cross-encoder call, then
    keeps each node's best score across all of them. Mutates node.score in place.

    An earlier version called reranker.postprocess_nodes(nodes, None, q) once per query
    in a loop -- correct, but confirmed live to add ~3-5s to retrieval_stage_s per
    retry-triggered question (4 sequential small GPU calls instead of 1 larger one).
    sentence-transformers' CrossEncoder.predict() already batches internally, so scoring
    the full query x node cross-product in a single call is the same computation done
    efficiently instead of split across 4 separate Python/CUDA round-trips.
    """
    if not nodes or not queries:
        return
    contents = [n.node.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
    pairs = [(q, c) for q in queries for c in contents]
    scores = reranker._model.predict(pairs)

    best = [float("-inf")] * len(nodes)
    for qi in range(len(queries)):
        offset = qi * len(nodes)
        for ni in range(len(nodes)):
            s = float(scores[offset + ni])
            if s > best[ni]:
                best[ni] = s
    for node, s in zip(nodes, best):
        node.score = s


