import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.admin.routes import router as admin_router
from backend.chat_service import ChatService
from backend.config import settings, validate_startup_config
from backend.rag.ingestion import load_index
from backend.rag.models import init_models
from backend.rag.retrieval import build_fusion_retriever, build_reranker
from backend.state import app_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory only (per PRD §7: "IP-based rate limiting stores IPs in memory only,
# not persisted") — counts reset on restart, which is fine for a single-process prototype.
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_startup_config()
    init_models()
    app_state["index"] = load_index()
    if app_state["index"] is None:
        logger.warning("No index found. Run scripts/seed_kb.py to ingest documents first.")
    else:
        app_state["fusion_retriever"] = build_fusion_retriever(app_state["index"])
        app_state["reranker"] = build_reranker()
    yield


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(admin_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    # Prior turns of this conversation, oldest first. Only the last
    # settings.max_history_messages are ever actually used (see
    # backend/rag/generation.py's _recent_history) -- the length cap here is just cheap
    # defense against an oversized payload, not the real cost control.
    history: list[HistoryMessage] = Field(default_factory=list, max_length=50)


class FeedbackRequest(BaseModel):
    message: str
    answer: str
    helpful: bool


async def _check_groq_reachable() -> bool:
    """Lightweight reachability probe: lists models rather than running a completion, so
    this never consumes generation tokens or counts against the chat rate limit -- with
    today's repeated daily-quota exhaustion fresh in mind, a health check must not itself
    be a source of token spend. Opt-in via /health?deep=true only (see health()), never
    run on a plain /health call, since a monitoring tool polling that endpoint every few
    seconds would otherwise add a constant background load of real Groq API calls for no
    operational benefit.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            )
            return response.status_code == 200
    except httpx.HTTPError:
        return False


@app.get("/health")
async def health(deep: bool = False):
    """IMPROVEMENTS.md #4.3: previously just {"status": "ok", "index_loaded": ...} --
    always "ok" even when the index hadn't loaded, so this couldn't actually distinguish
    a healthy server from a degraded one. Now reports GPU/device state (a silent CUDA ->
    CPU embedding fallback is 10-15x slower and previously only visible in logs) and
    whether GROQ_API_KEY is configured, and folds all of it into `status`. `deep=true`
    adds a real Groq network reachability check (see _check_groq_reachable) -- opt-in,
    not on every call, to keep this endpoint itself cheap and quota-free by default.
    """
    index_loaded = app_state.get("index") is not None
    groq_api_key_configured = bool(settings.groq_api_key.strip())
    gpu_available = torch.cuda.is_available()

    body = {
        "index_loaded": index_loaded,
        "gpu_available": gpu_available,
        "embedding_device": app_state.get("embedding_device"),
        "reranker_device_configured": settings.reranker_device,
        "groq_api_key_configured": groq_api_key_configured,
    }

    if deep:
        body["groq_reachable"] = groq_api_key_configured and await _check_groq_reachable()

    healthy = index_loaded and groq_api_key_configured and (not deep or body["groq_reachable"])
    body["status"] = "ok" if healthy else "degraded"
    return body


@app.get("/config/starter-questions")
async def starter_questions():
    return json.loads(settings.starter_questions_path.read_text(encoding="utf-8"))


@app.get("/avatar/{username}")
async def get_avatar(username: str):
    # Path(username).name strips any directory components, same guard as /documents/{filename}.
    safe_username = Path(username).name
    if settings.avatar_dir.exists():
        matches = sorted(settings.avatar_dir.glob(f"{safe_username}.*"))
        if matches:
            return FileResponse(matches[0])
    raise HTTPException(status_code=404, detail="No avatar set")


@app.get("/documents/{filename}")
async def get_document(filename: str):
    # Path(filename).name strips any directory components (e.g. "../../.env") so this
    # can only ever resolve to a file directly inside documents_dir.
    path = settings.documents_dir / Path(filename).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(path, filename=path.name)


@app.post("/chat")
@limiter.limit("30/hour")
async def chat(request: Request, body: ChatRequest):
    history = [{"role": h.role, "content": h.content} for h in body.history]
    service = ChatService(app_state)
    return StreamingResponse(
        service.stream(body.message, history), media_type="text/event-stream"
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": request.message,
        "answer": request.answer,
        "helpful": request.helpful,
    }
    with open(settings.feedback_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "ok"}
