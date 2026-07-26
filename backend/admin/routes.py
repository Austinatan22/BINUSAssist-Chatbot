import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.admin.auth import get_current_user, require_admin
from backend.admin.users import User, update_user
from backend.config import load_fallback_contacts, settings
from backend.rag.cache import clear_semantic_cache
from backend.rag.ingestion import (
    FACULTY_ROSTER_URL,
    SUPPORTED_EXTENSIONS,
    IngestionError,
    _cache_url_nodes,
    _faculty_records_to_nodes,
    add_document,
    build_index,
    delete_document_nodes,
    forget_scraped_url,
    forget_url_cache,
    load_documents,
    load_scraped_urls,
    record_scraped_url,
    refresh_faculty_snapshot,
    scrape_url,
    scrape_url_cached,
    validate_upload_content,
)
from backend.rag.retrieval import build_fusion_retriever, get_program_catalog
from backend.state import app_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_AVATAR_BYTES = 5 * 1024 * 1024
ALLOWED_AVATAR_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _rebuild_fusion_retriever() -> None:
    app_state["fusion_retriever"] = build_fusion_retriever(app_state["index"])


def _sync_index() -> None:
    """Rebuild BM25 from the live docstore and persist docstore.json so a restart sees the change."""
    _rebuild_fusion_retriever()
    app_state["index"].storage_context.persist(persist_dir=str(settings.chroma_persist_dir))
    # A cached answer (IMPROVEMENTS.md #3.1) is only valid for the KB state it was
    # generated against -- any document add/delete/URL-scrape invalidates all of it,
    # not just entries touching the changed document, since content elsewhere in the
    # answer could have cited it too. Called from every mutation route below except
    # reindex() (which doesn't call _sync_index and clears it separately).
    clear_semantic_cache()


@router.get("/documents")
async def list_documents():
    index = app_state.get("index")
    if index is None:
        return []

    collection = index.vector_store._collection
    result = collection.get(include=["metadatas"])

    docs: dict[str, dict] = {}
    for meta in result["metadatas"]:
        filename = meta.get("source_file")
        if filename is None:
            continue
        entry = docs.setdefault(
            filename, {"filename": filename, "chunk_count": 0, "ingested_at": None}
        )
        entry["chunk_count"] += 1
        if entry["ingested_at"] is None and meta.get("ingested_at"):
            entry["ingested_at"] = meta["ingested_at"]

    # Staleness flag (IMPROVEMENTS.md #5.2): a nudge, not an enforcement -- tuition/
    # admission content and year-versioned catalogs are swapped manually with no
    # re-scrape schedule, so surface age in the admin UI instead of letting it go
    # unnoticed.
    now = datetime.now(timezone.utc)
    for entry in docs.values():
        stale = False
        if entry["ingested_at"]:
            age_days = (now - datetime.fromisoformat(entry["ingested_at"])).days
            stale = age_days > settings.staleness_threshold_days
        entry["stale"] = stale

    return sorted(docs.values(), key=lambda d: d["filename"])


_YEAR_SUFFIX_RE = re.compile(r"_\d{4}$")


def _display_name_for_upload(filename: str) -> str:
    """"Computer_Science_2027.pdf" -> "Computer Science" -- same derivation as
    backend/rag/generation.py's _display_name_from_source_file and
    backend/rag/retrieval.py's get_program_catalog (duplicated rather than imported,
    per those modules' own stated rationale: a two-line regex isn't worth a
    cross-module dependency). Used here only to detect a same-program collision on
    upload (IMPROVEMENTS.md #5.3), e.g. a new catalog year alongside the old one --
    a coarse heuristic keyed on filename, not content, so an unrelated document that
    happens to reduce to the same stripped name would also be flagged; the cost of a
    false positive is just an extra confirmation step (supersede=true), never silent
    data loss, so this stays intentionally simple rather than trying to compare content.
    """
    stem = _YEAR_SUFFIX_RE.sub("", Path(filename).stem)
    return re.sub(r"_+", " ", stem).strip()


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile, supersede: bool = False):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 20MB limit")

    # Content validation (IMPROVEMENTS.md #8.4): extension/size alone don't catch a
    # mislabeled or malformed file. Runs before any of the supersede/delete logic below
    # so a bad upload is rejected with zero side effects on the existing KB.
    try:
        validate_upload_content(content, suffix)
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    index = app_state["index"]
    dest = settings.documents_dir / file.filename

    # Exact-filename re-upload (IMPROVEMENTS.md #5.3): unambiguous intent to replace,
    # not add -- without this, the old chunks stayed indexed under the identical
    # source_file value alongside the new ones, silently duplicating (and confusingly
    # dating) that document's content. No confirmation needed: re-uploading the exact
    # same filename is never ambiguous the way a differently-named collision is below.
    if dest.exists():
        delete_document_nodes(index, file.filename)

    # Different-filename collision (e.g. a new catalog year alongside the old one) --
    # NOT auto-replaced like the exact-filename case above: an admin could genuinely
    # want both versions coexisting for a transition period, so this requires an
    # explicit supersede=true rather than guessing.
    superseded_filename = None
    new_display_name = _display_name_for_upload(file.filename)
    existing_filename = get_program_catalog(index).get(new_display_name)
    if existing_filename and existing_filename != file.filename:
        if not supersede:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": (
                        f"'{existing_filename}' looks like the same program "
                        f"('{new_display_name}'). Replace it, or keep both by renaming "
                        "this file so it doesn't collide."
                    ),
                    "conflicting_filename": existing_filename,
                },
            )
        delete_document_nodes(index, existing_filename)
        old_path = settings.documents_dir / existing_filename
        if old_path.exists():
            old_path.unlink()
        superseded_filename = existing_filename

    dest.write_bytes(content)

    try:
        nodes = add_document(dest)
    except IngestionError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))

    if not nodes:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No extractable text found in file")

    index.insert_nodes(nodes)
    _sync_index()

    response = {"filename": file.filename, "chunks_added": len(nodes)}
    if superseded_filename:
        response["superseded_filename"] = superseded_filename
    return response


class UrlRequest(BaseModel):
    url: str


@router.post("/documents/url", status_code=status.HTTP_201_CREATED)
async def add_url(request: UrlRequest):
    """Scrape and index a web page. Persisted to scraped_urls.json so a future
    /admin/reindex re-fetches it instead of dropping it (IMPROVEMENTS.md #5.1)."""
    nodes = scrape_url(request.url)
    if not nodes:
        raise HTTPException(status_code=400, detail="Could not extract any text from that URL")

    app_state["index"].insert_nodes(nodes)
    record_scraped_url(request.url)
    _cache_url_nodes(request.url, nodes)  # seed last-known-good so a later reindex can't lose it
    _sync_index()
    return {"url": request.url, "chunks_added": len(nodes)}


@router.delete("/documents")
async def delete_document(filename: str):
    # filename is a query param, not a path param: URL-sourced "documents" contain
    # slashes (e.g. https://...) which don't fit cleanly into a single path segment.
    index = app_state.get("index")
    if index is None:
        raise HTTPException(status_code=404, detail="No index loaded")

    deleted = delete_document_nodes(index, filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    path = settings.documents_dir / filename
    if path.exists():
        path.unlink()
    forget_scraped_url(filename)  # no-op if filename isn't a known URL source
    forget_url_cache(filename)    # drop last-known-good so a re-add can't resurrect stale content

    _sync_index()
    return {"filename": filename, "chunks_deleted": deleted}


@router.post("/reindex")
async def reindex():
    """Rebuilds from backend/documents/ on disk AND re-fetches every URL previously
    added via /admin/documents/url (IMPROVEMENTS.md #5.1) -- a reindex used to only
    ever re-walk the documents dir, silently dropping every scraped page.

    A URL that fails to re-fetch (page moved, network blip, restyled HTML, rate-limit) is
    NOT silently lost: scrape_url_cached restores that URL's last-known-good chunks from the
    cache, so its content degrades to stale-but-present rather than vanishing from the
    rebuilt index. `urls_from_cache` reports which URLs were served stale (so an admin can
    investigate/re-add), and `urls_failed` is now only the URLs that have NEVER scraped
    successfully (no live result and nothing cached) -- the only ones genuinely absent."""
    nodes = load_documents(settings.documents_dir)

    failed_urls = []
    stale_urls = []
    for url in load_scraped_urls():
        url_nodes, from_cache = scrape_url_cached(url)
        if url_nodes:
            nodes.extend(url_nodes)
            if from_cache:
                stale_urls.append(url)
        else:
            failed_urls.append(url)
            logger.warning(
                "Reindex: %s could not be re-fetched and had no cached content", url
            )

    app_state["index"] = build_index(nodes)
    _rebuild_fusion_retriever()
    clear_semantic_cache()
    return {
        "chunks_indexed": len(nodes),
        "urls_failed": failed_urls,
        "urls_from_cache": stale_urls,
    }


@router.post("/faculty/refresh")
async def refresh_faculty():
    """Re-crawl the faculty roster + campus pages + scholar API, overwrite the cached
    snapshot, and rebuild the faculty nodes in the live index. This is the ONLY action that
    touches those fragile external sources -- routine /admin/reindex rebuilds faculty from
    the cached snapshot OFFLINE. Guarded: if the fresh crawl comes back notably smaller than
    the cache (a broken/rotated source), the snapshot is kept and this returns 409 without
    changing anything, so a degraded scrape can never silently shrink the roster."""
    result = refresh_faculty_snapshot()
    if not result["wrote"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Faculty refresh skipped: {result['reason']}. Cached snapshot left in place.",
        )
    index = app_state["index"]
    delete_document_nodes(index, FACULTY_ROSTER_URL)
    nodes = _faculty_records_to_nodes(result["records"], FACULTY_ROSTER_URL)
    index.insert_nodes(nodes)
    record_scraped_url(FACULTY_ROSTER_URL)
    _sync_index()
    return {"records": len(result["records"]), "nodes_indexed": len(nodes)}


class StarterQuestionsRequest(BaseModel):
    questions: list[str]


@router.put("/starter-questions")
async def update_starter_questions(request: StarterQuestionsRequest):
    questions = [q.strip() for q in request.questions if q.strip()]
    if not questions:
        raise HTTPException(status_code=400, detail="At least one question is required")

    settings.starter_questions_path.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"questions": questions}


class FallbackContact(BaseModel):
    role: str
    name: str
    email: str
    whatsapp: str = ""


class FallbackContactsRequest(BaseModel):
    contacts: list[FallbackContact]


@router.get("/fallback-contacts")
async def get_fallback_contacts():
    return load_fallback_contacts()


@router.put("/fallback-contacts")
async def update_fallback_contacts(request: FallbackContactsRequest):
    if not request.contacts:
        raise HTTPException(status_code=400, detail="At least one contact is required")

    contacts = []
    for c in request.contacts:
        role, name, email = c.role.strip(), c.name.strip(), c.email.strip()
        if not (role and name and email):
            raise HTTPException(
                status_code=400, detail="Role, name, and email are required for every contact"
            )
        contacts.append({"role": role, "name": name, "email": email, "whatsapp": c.whatsapp.strip()})

    settings.fallback_contacts_path.write_text(
        json.dumps(contacts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # A cached fallback/service-error answer (IMPROVEMENTS.md #3.1) embeds the contact
    # list at the time it was generated -- an edit here must invalidate it immediately,
    # same reasoning as get_fallback_message reading this file fresh on every call
    # rather than caching it itself.
    clear_semantic_cache()
    return {"contacts": contacts}


@router.get("/profile")
async def get_profile(user: User = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}


class ProfileUpdateRequest(BaseModel):
    username: str | None = None
    new_password: str | None = None


@router.put("/profile")
async def update_profile(request: ProfileUpdateRequest, user: User = Depends(get_current_user)):
    """Updates the current account's own username/password — not account creation."""
    username = request.username.strip() if request.username is not None else None
    if request.username is not None and not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if request.new_password is not None and len(request.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    try:
        updated = update_user(user.username, new_username=username, new_password=request.new_password or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"username": updated.username}


@router.post("/avatar", status_code=status.HTTP_201_CREATED)
async def upload_avatar(file: UploadFile, user: User = Depends(get_current_user)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_AVATAR_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {suffix}")

    content = await file.read()
    if len(content) > MAX_AVATAR_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds 5MB limit")

    settings.avatar_dir.mkdir(parents=True, exist_ok=True)
    for existing in settings.avatar_dir.glob(f"{user.username}.*"):
        existing.unlink()
    (settings.avatar_dir / f"{user.username}{suffix}").write_bytes(content)

    return {"status": "ok"}


@router.delete("/avatar")
async def delete_avatar(user: User = Depends(get_current_user)):
    if settings.avatar_dir.exists():
        for existing in settings.avatar_dir.glob(f"{user.username}.*"):
            existing.unlink()
    return {"status": "ok"}
