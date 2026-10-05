import json
import os
import re
import asyncio
import time
import struct
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, AsyncIterator
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from utils.groq_client import groq_client, TEXT_MODEL, JSON_CONFIG
from utils.json_repair import repair_json
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()
import hashlib
import tempfile
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from typing import List as _List
from rag import (
    retrieve_with_metadata_filter,
    normalize_query,
    RAG_TOP_K,
    RAG_MIN_SCORE,
)
from agent.validation import InvalidRepairPlan
from pipeline_orchestrator import run_full_pipeline, resolve_machine_from_query
from db_lock import LOCK_SCORE_THRESHOLD
from agent.models import CreateSessionRequest, CreateSessionResponse, AgentNextRequest
from agent import session_manager, repair_agent
from services.transcription_service import transcribe_audio_with_gemini, transcribe_audio_full
from services.inspect_part_service import inspect_part_service
from services.verification_service import verify_step_with_gemini
from services.machine_detection_service import detect_machine, detect_machine_from_frames, load_clip_model
from utils.helpers import (
    generate_cache_key,
    get_cached_response,
    cache_response,
    cleanup_old_files,
    derive_part_and_area,
)
from utils.machine_registry import (
    list_supported_machines,
    get_profile,
    resolve_machine_id,
)
from services.diagnosis_service import generate_diagnosis_with_gemini, _normalize_status

# ── Security layer ────────────────────────────────────────────────────────────
from security import (
    limiter,
    rate_limit_exceeded_handler,
    can_call_gemini,
    record_gemini_call,
    validate_video_upload,
    validate_audio_upload,
    verify_app_key,
    gemini_with_timeout,
    check_prompt_injection,
    log_security_event,
    sanitize_prompt_object,
    get_gemini_usage,
    # Limit constants — used in /health and startup log so operators can
    # verify what the server is enforcing without reading source code.
    VIDEO_MAX_BYTES,
    AUDIO_MAX_BYTES,
    VIDEO_MAX_SECONDS,
    AUDIO_MAX_SECONDS,
    GEMINI_HOURLY_LIMIT,
)

# ── Embedding wrapper ─────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _cached_embed_query(model: str, text: str) -> tuple:
    """
    Cache up to 256 query embeddings in memory (process-scoped, workers=1).

    Returns a tuple so lru_cache can hash the result.
    The caller converts back to list before use.

    Eviction: LRU — oldest unused entry is dropped when maxsize is reached.
    Thread-safety: lru_cache is thread-safe in CPython; safe under asyncio.
    """
    result = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_query",
    )
    return tuple(result["embedding"])

def normalize_diagnosis_response(diagnosis: dict, machine_type: str) -> dict:
    """
    Ensure every diagnosis response has machine_type at the top level
    AND inside solution. Uses setdefault() so existing values are never
    overwritten — only fills missing fields.
    
    Call this from every endpoint that returns a diagnosis:
      /diagnose, /diagnose/stream, /evaluate_text_rag
    """
    diagnosis.setdefault("machine_type", machine_type)
    
    solution = diagnosis.get("solution")
    if isinstance(solution, dict):
        solution.setdefault("machine_type", machine_type)
    
    return diagnosis


class GoogleEmbeddingsV1(Embeddings):
    def __init__(self, model: str, google_api_key: str):
        self.model = model
        genai.configure(api_key=google_api_key)

    def embed_documents(self, texts: _List[str]) -> _List[_List[float]]:
        # Called only during build — no cache needed here
        return [
            genai.embed_content(model=self.model, content=t, task_type="retrieval_document")["embedding"]
            for t in texts
        ]

    def embed_query(self, text: str) -> _List[float]:
        # Returns cached embedding on repeated queries — zero API cost on hit
        return list(_cached_embed_query(self.model, text))


# ============================================================================
# INITIALIZATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")
if not GOOGLE_AI_API_KEY:
    raise ValueError("❌ GOOGLE_AI_API_KEY not found! Set it in .env or HF Secrets")

genai.configure(api_key=GOOGLE_AI_API_KEY)
logger.info("✅ Gemini API configured successfully")

# ============================================================================
# CONFIGURATION
# ============================================================================

UPLOAD_DIR = Path("temp_uploads")
KB_DIR = Path("knowledge_base")
CACHE_DIR = Path("response_cache")
CHROMA_DIR = Path("chroma_db")
PLAN_CACHE_DIR = Path("plan_cache")   # ← repair plan cache
MAX_IMAGE_SIZE = 512
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_CACHE_AGE = 86400
PLAN_CACHE_TTL = int(os.environ.get("PLAN_CACHE_TTL_SECONDS", str(30 * 24 * 3600)))
PLAN_CACHE_MAX_ENTRIES = int(os.environ.get("PLAN_CACHE_MAX_ENTRIES", "500"))
PLAN_CACHE_EPHEMERAL = os.environ.get("PLAN_CACHE_EPHEMERAL", "auto").lower()
THREAD_POOL_WORKERS = int(os.getenv("AGRIFIX_THREAD_POOL_WORKERS", "20"))

UPLOAD_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
PLAN_CACHE_DIR.mkdir(exist_ok=True)
EFFECTIVE_PLAN_CACHE_TTL: int = PLAN_CACHE_TTL      # may be reduced for ephemeral storage
PLAN_CACHE_IS_EPHEMERAL: bool = False               # detected at startup

def _safe_filename(raw: str, fallback: str) -> str:
    """
    Strip any directory components from a client-supplied filename.
    Prevents path traversal attacks like '../../../etc/passwd'.
    Returns only the final filename component, or `fallback` if empty.

    Example:
        _safe_filename("../../../etc/passwd", "rec.mp4") → "etc/passwd" → "passwd"
        Wait — Path("../../../etc/passwd").name → "passwd"  ✓
    """
    name = Path(raw).name if raw else ""
    # Also strip leading dots that could create hidden files (e.g. ".bashrc")
    name = name.lstrip(".")
    return name or fallback

# ============================================================================
# REPAIR PLAN CACHE
# ============================================================================
# Key: SHA-256( machine_type + "|" + problem_cluster )
# Value: JSON file on disk — survives server restarts.
#
# problem_cluster = sorted(infer_problem_categories(problem_text)).join(",")
# This normalises "motor won't start" and "engine not starting" to the same
# cluster key, so we reuse the cached plan for semantically identical complaints.
#
# Cache is intentionally NOT invalidated by RAG chunk updates — if the problem
# cluster is the same, the repair logic is the same.  TTL is 30 days.
# Operators can clear the cache folder to force fresh generation.

def _plan_cache_key(machine_type: str, problem_text: str,
                    visual_hash: str = "") -> str:
    """
    Deterministic cache key: machine_type + normalized query + visual hash.

    The visual_hash is a perceptual hash of the best video frame (8 bytes, hex).
    This is CRITICAL for accuracy: without it, "water_pump + not_working" would
    serve the same cached plan whether the motor is dead OR water is flowing.
    """
    cluster = normalize_query(problem_text)[:60]
    raw = f"{machine_type.lower()}|{cluster}|{visual_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _phash_bytes(image_bytes: bytes, hash_size: int = 8) -> str:
    """
    Average perceptual hash — 16-char hex string used in the plan cache key.
    Operates on JPEG bytes already in RAM from detection.frames. No I/O.

    BUG FIX (2026-03-12): the original implementation computed a 1024-bit
    integer (32×32 pixels) then called .to_bytes(8, "big"), which always
    raised OverflowError — silently caught, always returning "00000000".
    This meant every cache key ended with "|00000000", so the plan cache
    never differentiated by visual content. Fix: take only the first 64 bits
    of the bit string (8 bytes) — sufficient for cache differentiation across
    the small machine/symptom space (<<2^64 unique visual states in practice).
    """
    import struct as _struct
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
        pixels = list(img.getdata())
        mean = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= mean else "0" for p in pixels)
        # Take first 64 bits only → packs cleanly into 8 bytes, no OverflowError
        return _struct.pack(">Q", int(bits[:64], 2)).hex()
    except Exception:
        return "00000000"


def _plan_cache_get(key: str) -> Optional[Dict[str, Any]]:
    """Return cached plan dict or None if missing / expired."""
    p = PLAN_CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > EFFECTIVE_PLAN_CACHE_TTL:
            p.unlink(missing_ok=True)
            logger.info(f"🗑️  Plan cache expired and removed: {key}")
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        logger.info(f"🎯 Plan cache HIT: {key}")
        return data
    except Exception as exc:
        logger.warning(f"Plan cache read error ({key}): {exc}")
        return None


def _plan_cache_set(key: str, plan: Dict[str, Any]) -> None:
    """Write plan to disk cache. Evicts oldest entries when over limit. Failures are non-fatal."""
    try:
        # P2-3: Evict oldest files when cache exceeds max entries
        _existing = sorted(
            PLAN_CACHE_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(_existing) >= PLAN_CACHE_MAX_ENTRIES:
            _oldest = _existing.pop(0)
            try:
                _oldest.unlink()
            except FileNotFoundError:
                pass
            logger.debug("🗑️  Plan cache evicted (max entries %d): %s", PLAN_CACHE_MAX_ENTRIES, _oldest.name)

        p = PLAN_CACHE_DIR / f"{key}.json"
        p.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        logger.info(f"💾 Plan cache WRITE: {key}")
    except Exception as exc:
        logger.warning(f"Plan cache write error ({key}): {exc}")

# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global EFFECTIVE_PLAN_CACHE_TTL, PLAN_CACHE_IS_EPHEMERAL

    logger.info("AgriFix v4.1 Backend starting...")
    UPLOAD_DIR.mkdir(exist_ok=True)
    KB_DIR.mkdir(exist_ok=True)

    # P2-3: Detect ephemeral storage and compute effective cache TTL.
    # Runtime detection using tempfile.gettempdir() — no hardcoded paths.
    # Operators can override with PLAN_CACHE_EPHEMERAL=true|false|auto.
    if PLAN_CACHE_EPHEMERAL == "true":
        EFFECTIVE_PLAN_CACHE_TTL = min(PLAN_CACHE_TTL, 86400)
        PLAN_CACHE_IS_EPHEMERAL = True
        logger.info(
            "💾 Plan cache: ephemeral (env override) — TTL capped at %d hours",
            EFFECTIVE_PLAN_CACHE_TTL // 3600,
        )
    elif PLAN_CACHE_EPHEMERAL == "false":
        EFFECTIVE_PLAN_CACHE_TTL = PLAN_CACHE_TTL
        PLAN_CACHE_IS_EPHEMERAL = False
        logger.info("💾 Plan cache: persistent (env override) — TTL %d days", PLAN_CACHE_TTL // 86400)
    else:  # "auto"
        _temp_root = Path(tempfile.gettempdir()).resolve()
        _cache_path = PLAN_CACHE_DIR.resolve()
        PLAN_CACHE_IS_EPHEMERAL = (
            _cache_path == _temp_root
            or _temp_root in _cache_path.parents
        )
        if PLAN_CACHE_IS_EPHEMERAL and PLAN_CACHE_TTL > 86400:
            EFFECTIVE_PLAN_CACHE_TTL = 86400
            logger.warning(
                "⚠️  Plan cache on ephemeral storage (%s) — TTL reduced from %d days to %d hours. "
                "Set PLAN_CACHE_EPHEMERAL=false or PLAN_CACHE_TTL_SECONDS to override.",
                _cache_path, PLAN_CACHE_TTL // 86400, EFFECTIVE_PLAN_CACHE_TTL // 3600,
            )
        else:
            EFFECTIVE_PLAN_CACHE_TTL = PLAN_CACHE_TTL
            logger.info(
                "💾 Plan cache: %s — TTL %d days",
                "ephemeral" if PLAN_CACHE_IS_EPHEMERAL else "persistent",
                EFFECTIVE_PLAN_CACHE_TTL // 86400,
            )

    default_executor = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)
    loop = asyncio.get_running_loop()
    loop.set_default_executor(default_executor)
    logger.info("Thread pool configured with %d workers", THREAD_POOL_WORKERS)

    try:
        # Load CLIP once at startup — shared across all requests (workers=1)
        clip_ok = await asyncio.to_thread(load_clip_model)
        logger.info(
            "CLIP detector: ACTIVE" if clip_ok
            else "⚠️ CLIP detector: DISABLED — audio keywords + Gemini fallback only"
        )

        from utils.machine_registry import _init_machine_categories
        _init_machine_categories()
        logger.info("Machine categories initialized for shutdown instructions")

        from crossencoder_reranker import _RERANKER
        await asyncio.to_thread(_RERANKER._load)
        logger.info("CrossEncoder pre-loaded")

        rag_ok = load_vector_db()
        logger.info("RAG pipeline: ACTIVE" if rag_ok else "⚠️  RAG pipeline: DISABLED — Gemini-only mode")
        yield
    finally:
        logger.info("AgriFix Backend shutting down...")
        default_executor.shutdown(wait=True)
        cleanup_old_files(UPLOAD_DIR)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AgriFix AI Backend - Hybrid Production",
    description="Agricultural machinery repair assistant with AI-powered stateful agent",
    version="4.2.0",
    lifespan=lifespan,
)

# ── Security: attach SlowAPI limiter state ────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

from agent.validation import InvalidRepairPlan

@app.exception_handler(InvalidRepairPlan)
async def invalid_repair_plan_handler(request: Request, exc: InvalidRepairPlan):
    """
    A repair plan is structurally broken — this is a BACKEND DEFECT,
    not a mechanical fault with the farmer's machine.
    
    Return a generic "service error" response, NEVER an escalation card
    telling the farmer to "consult a certified mechanic." Nothing is wrong
    with their equipment; something is wrong with this service.
    """
    logger.critical(
        "❌ InvalidRepairPlan reached API boundary: %s | path=%s",
        exc, request.url.path,
    )
    # TODO: send to Sentry/alerting here
    
    return JSONResponse(
        status_code=503,
        content={
            "status": "service_error",
            "message": "Something went wrong while preparing your repair guide. "
                       "Please try again in a moment. If the problem continues, "
                       "restart the app.",
            "message_hi": "आपकी मरम्मत गाइड तैयार करते समय कुछ गलत हो गया। "
                          "कृपया कुछ क्षण बाद पुनः प्रयास करें।",
        },
    )

app.add_middleware(
    CORSMiddleware,
    # ⚠️  Never use "*" with allow_credentials=True — browsers block it and it
    # allows any website to make credentialed cross-origin requests to this API.
    # List only the exact origins that legitimately call this backend.
    # Flutter mobile apps do NOT use CORS (no browser), so this only matters
    # for web builds or the HuggingFace Spaces demo page.
    allow_origins=[
        "https://agrifix.hf.space",        # HuggingFace Spaces demo (update to your Space URL)
        "http://localhost:8000",            # local dev
        "http://localhost:3000",            # local web dev
    ],
    allow_credentials=False,               # no cookies/sessions — app key is in headers
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-App-Key"],
)

# ============================================================================
# RAG — VECTOR DATABASE
# ============================================================================

vector_db: Chroma = None


def load_vector_db() -> bool:
    global vector_db
    if not CHROMA_DIR.exists():
        logger.warning("⚠️  chroma_db/ not found — RAG disabled")
        return False
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vector_db = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )
        # NEW: load machine aliases + symptoms into router
        from query_router import load_machine_registry
        load_machine_registry(vector_db)
        from rag import load_dynamic_compat_map
        load_dynamic_compat_map(vector_db)
        stored_ids = vector_db.get()["ids"]
        logger.info(f"✅ Chroma DB loaded — {len(stored_ids)} chunks")
        return True
    except Exception as exc:
        logger.error(f"❌ Failed to load Chroma DB: {exc}")
        return False

def retrieve_rag_context(query: str, machine_type: str, k: int = RAG_TOP_K) -> str:
    """
    Backward-compatible wrapper for the optimized v7.0 RAG pipeline.
    Category inference has been removed in favor of pure semantic search.
    """
    if vector_db is None:
        return ""

    return retrieve_with_metadata_filter(
        vector_db=vector_db,
        query=query,
        machine_type=machine_type,
        k=k,
    )


# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

def load_knowledge_base(machine_type: str) -> str:
    knowledge_file = KB_DIR / f"{machine_type.lower()}_facts.txt"
    if knowledge_file.exists():
        try:
            return knowledge_file.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error(f"Error loading knowledge base: {exc}")
    return "Use general agricultural machinery repair knowledge."


# ============================================================================
# SAFETY CHECK (local — kept in main to avoid dependency on PIL in services)
# ============================================================================

async def safety_check_with_gemini(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)

        prompt = """You are a safety camera watching over a farmer inspecting their farm machinery or equipment.
The machine could be a tractor, harvester, thresher, pump, electric motor, chaff cutter, power tiller, generator, or any other farm equipment.
Look at this image for any immediate danger — regardless of machine type.

Check for ALL of these hazards:
1. Hands, fingers, or clothing too close to moving parts — fan blades, belts, chains, drum, cutter blades, rotating shaft
2. Farmer working under a machine that is NOT safely supported
3. Fire, smoke, sparks, or glowing near fuel, oil, or electrical components
4. Fuel or oil actively leaking or spraying
5. Engine or motor running (fan spinning, exhaust smoke) when it should be switched off
6. Exposed live electrical wires, open junction boxes near hands — for electric machines
7. Farmer standing directly in front of a feed inlet or discharge chute of thresher / chaff cutter
8. No safety guard on chaff cutter or thresher feed opening

Return ONLY this JSON:
{
  "safe": true | false,
  "hazard_detected": "Plain description of the danger in simple words a farmer understands, or null if safe",
  "severity": "low" | "medium" | "high" | null,
  "warning_message": "Short urgent instruction — e.g. 'STOP! Move your hand away from the spinning belt!' or 'Switch off the power immediately — a live wire is exposed!' Null if safe."
}
If ANY hazard is detected, set safe to false immediately. Do not wait for certainty.
Return ONLY JSON, no markdown."""

        # VISION CALL — kept on Gemini until vision migration task  # MIGRATED: Gemini → Groq (pending)
        # Future: swap to llama-3.2-11b-vision-preview when visual migration runs.
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = await asyncio.to_thread(
            lambda: model.generate_content([prompt, image])
        )
        result = repair_json(response.text)  # MIGRATED: Gemini → Groq (repair_json replaces json.loads+sanitize)
        if not result.get("safe"):
            logger.warning(f"⚠️ Safety hazard: {result.get('hazard_detected')}")
        return result
    except Exception as exc:
        logger.exception("❌ Safety check unavailable — failing closed")
        return {
            "safe": False,
            "analysis_available": False,
            "hazard_detected": "Unable to verify scene safety — analysis unavailable.",
            "severity": "high",
            "warning_message": (
                "STOP. The safety camera could not analyze the scene. "
                "Do not continue until the area has been visually checked."
            ),
            "error_code": "SAFETY_ANALYSIS_UNAVAILABLE",
        }


# ============================================================================
# SSE HELPER
# ============================================================================

def _sse(event: str, data: dict) -> str:
    payload = json.dumps({"event": event, **data})
    return f"data: {payload}\n\n"


def _local_fallback_diagnosis(machine_type: str, problem_text: str) -> dict:
    """
    Minimal fallback returned when Gemini is unavailable (budget exceeded / timeout).
    Returns a valid response shape so Flutter doesn't crash.
    """
    from utils.machine_registry import get_profile_or_default, get_safety_warnings
    profile = get_profile_or_default(machine_type)
    return {
        "status": "error",
        "problem_description": problem_text,
        "technical_analysis": "Diagnosis service temporarily unavailable.",
        "solution": {
            "status": "error",
            "machine_type": machine_type,
            "problem_identified": problem_text,
            "steps": [],
            "safety_warnings_en": get_safety_warnings(machine_type, "en") or ["Consult a certified mechanic."],
            "safety_warnings_hi": get_safety_warnings(machine_type, "hi") or ["प्रमाणित मैकेनिक से संपर्क करें।"],
            "tools_needed": [],
        },
        "rag_source": "unavailable",
        "machine_label": profile.label_en if profile else machine_type,
    }


# ============================================================================
# API ENDPOINTS — EXISTING (unchanged interface)
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "operational",
        "service": "AgriFix Production Backend",
        "version": "4.1.0",
        "architecture": {
            "machine_detection": "CLIP zero-shot + audio keywords (local) + Gemini fallback",
            "transcription": "Gemini 2.5 Flash",
            "diagnosis": "Gemini 2.5 Flash + RAG",
            "verification": "Gemini 2.5 Flash Vision",
            "agent": "Stateful repair agent (Gemini + safety rules)",
        },
    }


@app.get("/health")
async def health(request: Request):
    ip = request.client.host if request.client else "unknown"
    rag_chunks = 0
    if vector_db is not None:
        try:
            rag_chunks = vector_db._collection.count()
        except Exception:
            pass
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini": "configured",
        "knowledge_base": len(list(KB_DIR.glob("*.txt"))),
        "rag": {
            "status": "active" if vector_db is not None else "disabled",
            "chunks": rag_chunks,
            "top_k": RAG_TOP_K,
            "min_score": RAG_MIN_SCORE,
            "lock_threshold": LOCK_SCORE_THRESHOLD,
            "pipeline_version": "v10.1+phases123",
        },
        "active_sessions": len(session_manager.list_sessions()),
        "your_gemini_calls_this_hour": get_gemini_usage(ip),
        "plan_cache": {
            "entries": len(list(PLAN_CACHE_DIR.glob("*.json"))),
            "ttl_days": EFFECTIVE_PLAN_CACHE_TTL // 86400,
            "max_entries": PLAN_CACHE_MAX_ENTRIES,
            "ephemeral": PLAN_CACHE_IS_EPHEMERAL,
        },
        # Server-enforced limits — visible to operators; attackers already know
        # the client limits, so publishing server limits does not aid attacks.
        "limits": {
            "video_max_mb":        VIDEO_MAX_BYTES // 1_048_576,
            "audio_max_mb":        AUDIO_MAX_BYTES // 1_048_576,
            "video_max_seconds":   int(VIDEO_MAX_SECONDS),
            "audio_max_seconds":   int(AUDIO_MAX_SECONDS),
            "gemini_calls_per_ip_per_hour": GEMINI_HOURLY_LIMIT,
        },
    }


@app.get("/machines")
async def list_machines():
    """
    List all farm machines supported by AgriFixAR.
    Returns machine IDs, labels, categories, and recognised aliases.
    Unity / Flutter can use this to populate the machine picker and
    validate YOLO detection labels before sending to /diagnose.
    """
    return {
        "status": "ok",
        "supported_machines": list_supported_machines(),
        "total": len(list_supported_machines()),
    }


@app.get("/machines/{machine_type}")
async def get_machine_info(machine_type: str):
    """
    Return detailed info about a specific machine type including
    all parts, area zones, and safety warnings.
    """
    profile = get_profile(machine_type)
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Machine '{machine_type}' not recognised. Call GET /machines for the full list.",
        )
    return {
        "machine_id": profile.machine_id,
        "label_en": profile.label_en,
        "label_hi": profile.label_hi,
        "category": profile.category,
        "farmer_intro_en": profile.farmer_intro_en,
        "farmer_intro_hi": profile.farmer_intro_hi,
        "area_zones": [
            {
                "id": z.id,
                "label_en": z.label_en,
                "label_hi": z.label_hi,
                "farmer_description_en": z.farmer_description_en,
                "farmer_description_hi": z.farmer_description_hi,
            }
            for z in profile.area_zones
        ],
        "parts": [
            {
                "id": p.id,
                "area_zone": p.area_zone,
                "label_en": p.label_en,
                "label_hi": p.label_hi,
                "farmer_description_en": p.farmer_description_en,
                "ar_model": p.ar_model,
            }
            for p in profile.parts
        ],
        "safety_warnings_en": profile.base_safety_warnings_en,
        "safety_warnings_hi": profile.base_safety_warnings_hi,
    }


@app.post("/detect_machine")
@limiter.limit("10/minute")
async def detect_machine_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile = File(default=None),
    _auth: None = Depends(verify_app_key),
):
    """
    Stage-0 endpoint: detect machine type from video + optional audio.

    Flutter calls this BEFORE /diagnose/stream so the farmer can confirm
    the detected machine type on a dedicated confirmation screen.

    Response:
        {
          "machine_type":    "electric_water_pump",
          "confidence":      0.91,
          "source":          "clip",
          "label_en":        "Electric Water Pump",
          "label_hi":        "इलेक्ट्रिक वॉटर पंप",
          "alternatives":    [{"id": "diesel_pump", "label_en": "...", "label_hi": "..."}],
          "needs_confirmation": true   ← true when confidence < 0.80
        }

    Token cost: 0 (CLIP-only when confident, Gemini fallback only when needed).
    Latency: ~1–2 s on HF free tier.
    """
    ip = request.client.host if request.client else "unknown"
    logger.info(f"🔍 /detect_machine ip={ip}")
    request_id = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]

    try:
        video_bytes = await video.read()
        validate_video_upload(video.filename or "rec.mp4", video_bytes, ip=ip)

        _vname = _safe_filename(video.filename, "rec.mp4")
        video_path = UPLOAD_DIR / f"{request_id}_detect_{_vname}"
        video_path.write_bytes(video_bytes)
        background_tasks.add_task(lambda: video_path.unlink(missing_ok=True))

        transcription_text: Optional[str] = None
        if audio is not None:
            try:
                audio_bytes = await audio.read()
                if len(audio_bytes) > 512:
                    validate_audio_upload(audio.filename or "rec.m4a", audio_bytes, ip=ip)
                    _aname = _safe_filename(audio.filename, "rec.m4a")
                    audio_path = UPLOAD_DIR / f"{request_id}_detect_audio_{_aname}"
                    audio_path.write_bytes(audio_bytes)
                    background_tasks.add_task(lambda: audio_path.unlink(missing_ok=True))
                    if can_call_gemini(ip):
                        transcription_text = await gemini_with_timeout(
                            transcribe_audio_with_gemini(audio_path),
                            fallback=None, context="detect/transcription",
                        )
                        if transcription_text:
                            record_gemini_call(ip)
            except Exception:
                pass  # audio is optional — carry on without it

        detection = await detect_machine(
            video_path=video_path,
            transcription_text=transcription_text,
        )
        if detection.gemini_used:
            record_gemini_call(ip)

        resolved = detection.machine_type
        profile   = get_profile(resolved)

        # Build alternatives list from supported machines (exclude detected)
        all_machines = list_supported_machines()
        alternatives = [
            {
                "id":       m["machine_id"],
                "label_en": m["label_en"],
                "label_hi": m.get("label_hi", m["label_en"]),
            }
            for m in all_machines
            if m["machine_id"] != resolved
        ][:8]  # cap at 8 so the UI list stays manageable

        return JSONResponse(content={
            "machine_type":       resolved,
            "confidence":         round(detection.confidence, 3),
            "source":             detection.source,
            "clip_confidence":    round(detection.clip_confidence or 0.0, 3),
            "audio_confidence":   round(detection.audio_confidence or 0.0, 3),
            "label_en":           profile.label_en  if profile else resolved,
            "label_hi":           profile.label_hi  if profile else resolved,
            "needs_confirmation": detection.confidence < 0.80,
            "alternatives":       alternatives,
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ /detect_machine failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/diagnose")
@limiter.limit("5/minute")
async def diagnose(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    machine_type: str = Form(default=""),   # optional hint — auto-detected now
    language: str = Form(default="en"),
    _auth: None = Depends(verify_app_key),
):
    """
    Main diagnosis endpoint — response format UNCHANGED.
    """
    ip = request.client.host if request.client else "unknown"
    logger.info(f"📥 /diagnose hint={machine_type!r} lang={language} ip={ip}")
    request_id = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]

    try:
        if not audio.filename:
            raise HTTPException(status_code=400, detail="Audio file required")

        # Read bytes first so we can validate before touching disk
        audio_bytes = await audio.read()
        video_bytes = await video.read()

        # ── Security: file validation ─────────────────────────────────────────
        validate_audio_upload(audio.filename or "rec.m4a", audio_bytes, ip=ip)
        if video.filename and len(video_bytes) > 1024:
            validate_video_upload(video.filename, video_bytes, ip=ip)

        # Save audio
        _aname = _safe_filename(audio.filename, "rec.m4a")
        audio_path  = UPLOAD_DIR / f"{request_id}_audio_{_aname}"
        audio_path.write_bytes(audio_bytes)
        background_tasks.add_task(lambda: audio_path.unlink(missing_ok=True))

        # Save video only if meaningful content exists
        video_path: Optional[Path] = None
        if video.filename and len(video_bytes) > 1024:
            _vname = _safe_filename(video.filename, "rec.mp4")
            video_path = UPLOAD_DIR / f"{request_id}_video_{_vname}"
            video_path.write_bytes(video_bytes)
            background_tasks.add_task(lambda: video_path.unlink(missing_ok=True))

        # ── Token-budget architecture ─────────────────────────────────────────
        # Goal: exactly 2 Gemini calls per /diagnose request.
        #
        #   Call 1 — transcribe_audio_with_gemini  (audio → text)
        #   Call 2 — generate_diagnosis_with_gemini (reason once, output JSON)
        #
        # CLIP detection is local (CPU-only).  We run CLIP and transcription
        # concurrently so the combined wall-clock time is cut roughly in half.
        # detect_machine's internal Gemini fallback is bypassed by supplying the
        # transcription text on a second pass only if CLIP confidence is low —
        # the keyword matcher then resolves the machine type locally.
        # ─────────────────────────────────────────────────────────────────────

        # ── Transcription: two-stage Groq Whisper + Gemini normalisation ────────
        # transcribe_audio_full() returns {"raw_transcript", "normalized_problem"}.
        # raw_transcript = exact Whisper output (accent-accurate, no paraphrase).
        # normalized_problem = Gemini text normalisation (tiny text-only call).
        # Falls back to original Gemini audio path when GROQ_API_KEY is absent.
        # Mutable cells — closure-safe storage for async inner functions
        _raw_transcript_store:  list[str]  = [""]
        _transcript_meta_store: list[dict] = [{}]  # quality_grade, gemini_used, etc.

        async def _transcribe_batch() -> str:
            if can_call_gemini(ip):
                result = await gemini_with_timeout(
                    transcribe_audio_full(audio_path),   # machine_hint added below
                    fallback=None,
                    context="transcription",
                )
                record_gemini_call(ip)
                if result and isinstance(result, dict):
                    _raw_transcript_store[0]  = result.get("raw_transcript", "")
                    _transcript_meta_store[0] = result
                    return result.get("normalized_problem") or "farm machine problem"
                return "farm machine problem"
            logger.warning(f"⚠️  Gemini budget exceeded for {ip} — using fallback transcription")
            return "farm machine problem"

        async def _detect_clip_batch() -> Any:
            # CLIP-only first pass — no Gemini, no transcription text needed
            return await detect_machine(video_path=video_path, transcription_text=None)

        # Run CLIP inference and Gemini transcription concurrently
        problem_text_raw, detection_clip = await asyncio.gather(
            _transcribe_batch(), _detect_clip_batch()
        )

        # ── Security: prompt injection check ─────────────────────────────────
        problem_text = check_prompt_injection(
            problem_text_raw or "", ip=ip, field="transcription"
        )
        # Strip "unknown machine" hallucination from transcription normalizer
        problem_text = re.sub(
            r"^The\s+unknown\s+machine'?s?\s+",
            "The ",
            problem_text,
            flags=re.IGNORECASE,
        )
        problem_text = re.sub(
            r"^The\s+unknown\s+machines'?\s+",
            "The ",
            problem_text,
            flags=re.IGNORECASE,
        )
        # Also catch "on/in the unknown machine" and "unknown machine has/is"
        problem_text = re.sub(
            r"\s+(?:on|in)\s+the\s+unknown\s+machine",
            "",
            problem_text,
            flags=re.IGNORECASE,
        )
        problem_text = re.sub(
            r"^The\s+unknown\s+machine\s+(?:has|is)\s+",
            "The ",
            problem_text,
            flags=re.IGNORECASE,
        )

        # ── Machine-context re-validation (zero extra API calls) ──────────────
        # Now that CLIP has given us a machine_hint, validate the transcription
        # result with the correct context. Only triggers the re-check when
        # Gemini was skipped (fast path) — if Gemini already ran it already
        # had access to the hint via machine_hint="".
        # Cost: pure Python bucket check, <1 ms.
        _clip_hint_for_transcript = detection_clip.machine_type or ""
        _tmeta = _transcript_meta_store[0]
        if (
            _clip_hint_for_transcript
            and _tmeta.get("gemini_used") == "false"    # Gemini was skipped
            and _raw_transcript_store[0]                # we have a raw transcript
        ):
            from services.transcription_service import (
                _detect_symptom_buckets, _validate_output, _light_clean_raw
            )
            _raw_for_check = _raw_transcript_store[0]
            _buckets       = _detect_symptom_buckets(_raw_for_check)
            _ok, _reason   = _validate_output(
                _raw_for_check, problem_text, _buckets, _clip_hint_for_transcript
            )
            if not _ok:
                logger.warning(
                    f"⚠️  Post-CLIP validation failed ({_reason}) — "
                    f"re-normalising with machine_hint={_clip_hint_for_transcript!r}"
                )
                if can_call_gemini(ip):
                    from services.transcription_service import _extract_with_gemini
                    problem_text = await _extract_with_gemini(
                        _raw_for_check, _buckets, _clip_hint_for_transcript
                    )
                    record_gemini_call(ip)
                    problem_text = check_prompt_injection(
                        problem_text, ip=ip, field="context_revalidation"
                    )

        # ── Resolve machine type without an extra Gemini call ─────────────────
        resolved  = detection_clip.machine_type
        detection = detection_clip

        if detection_clip.confidence < 0.55:
            # CLIP uncertain: re-run locally with transcription keywords.
            # CRITICAL: pass video_path=None — video was already decoded above.
            # Restore frames from the first decode so diagnosis still gets them.
            detection = await detect_machine(
                video_path=None,          # ← no re-decode
                transcription_text=problem_text,
            )
            detection.frames = detection_clip.frames  # restore already-decoded frames
            if detection.gemini_used:
                record_gemini_call(ip)
            resolved = detection.machine_type

        # Client hint overrides when confidence is still low
        if machine_type.strip():
            hint = resolve_machine_id(machine_type.strip())
            if detection.confidence < 0.55 and get_profile(hint):
                logger.info(f"🔧 Low confidence ({detection.confidence:.2f}) — using client hint: {hint}")
                resolved = hint

        logger.info(
            f"🔧 Machine: {resolved}  "
            f"(detected={detection.machine_type} conf={detection.confidence:.2f} "
            f"src={detection.source} gemini={detection.gemini_used})"
        )

        # ── Reuse frames already decoded by CLIP — zero extra video I/O ──────
        # detection.frames holds the same early/mid/late JPEG bytes that CLIP
        # quality-scored and used for classification. No re-open, no re-decode.
        clip_frames  = detection.frames          # list[bytes], already in RAM
        mid_frame    = clip_frames[len(clip_frames) // 2] if clip_frames else None
        visual_hash  = _phash_bytes(mid_frame) if mid_frame else ""
        logger.info(f"📸 Frames from CLIP cache: {len(clip_frames)}/3  phash={visual_hash or 'none'}")

        # ── Plan cache check — avoids LLM call entirely on hit ────────────────
        plan_cache_key = _plan_cache_key(resolved, problem_text, visual_hash)
        cached_plan    = _plan_cache_get(plan_cache_key)

        # ── Phase 1–3 Pipeline: router → DB lock → visual gate ────────────────
        # Phase 1 (router) is skipped here because machine type is already
        # Use machine override only when CLIP confidence is high
        # When low, let the router determine machine type from query keywords
        _machine_override = resolved if detection.confidence >= 0.70 else None
            
        pipeline = await run_full_pipeline(
                query=problem_text,
                vector_db=vector_db,
                frame_bytes=mid_frame,
                language=language,
                machine_type_override=_machine_override,
            )

        if pipeline.blocked:
            # Phase 2 (no DB match) or Phase 3 (visual gate failed) — no LLM call
            logger.info(f"🔒 /diagnose pipeline blocked: {pipeline.block_reason}")
            block_response = pipeline.response
            block_response["request_id"]   = request_id
            block_response["transcription"] = problem_text
            block_response["detection"]     = {
                "machine_type":     resolved,
                "confidence":       detection.confidence,
                "source":           detection.source,
                "clip_confidence":  detection.clip_confidence,
                "audio_confidence": detection.audio_confidence,
                "gemini_used":      detection.gemini_used,
            }
            return JSONResponse(content=block_response)

        # All gates passed — safe to proceed with LLM diagnosis
        rag_context = pipeline.rag_context
        knowledge   = load_knowledge_base(pipeline.machine_type)

        # ── Diagnosis: ONE Gemini reasoning call (now multimodal 3-frame) ─────
        if cached_plan is not None:
            diagnosis = cached_plan
            diagnosis["cache_hit"] = True
            logger.info(f"🎯 /diagnose cache HIT key={plan_cache_key}")
        elif can_call_gemini(ip):
            diagnosis = await gemini_with_timeout(
                generate_diagnosis_with_gemini(
                    machine_type=pipeline.machine_type,
                    problem_text=problem_text,
                    language=language,
                    rag_context=rag_context,
                    knowledge_base=knowledge,
                    visual_frames=clip_frames,
                    top_score=round(pipeline.lock.score, 3) if pipeline.lock else 0.0,
                ),
                fallback=None,
                context="diagnosis",
            )
            if diagnosis:
                record_gemini_call(ip)
                # Enrich with visual gate findings before caching
                if pipeline.gate:
                    diagnosis["confirmed_part"]     = pipeline.gate.part_id
                    diagnosis["visual_observation"] = pipeline.gate.fault_description
                    diagnosis["rag_score"]          = round(pipeline.lock.score, 3)
                _plan_cache_set(plan_cache_key, diagnosis)
            else:
                diagnosis = _local_fallback_diagnosis(resolved, problem_text)
        else:
            logger.warning(f"⚠️  Gemini budget exceeded for {ip} — skipping diagnosis Gemini call")
            diagnosis = _local_fallback_diagnosis(resolved, problem_text)

        diagnosis["request_id"]      = request_id
        diagnosis["transcription"]    = problem_text
        diagnosis["raw_transcript"]   = _raw_transcript_store[0] or problem_text
        _tmeta_d = _transcript_meta_store[0]
        diagnosis["transcription_meta"] = {
            "quality_grade": _tmeta_d.get("quality_grade", ""),
            "gemini_used":   _tmeta_d.get("gemini_used", ""),
            "skip_reason":   _tmeta_d.get("skip_reason", ""),
        }
        diagnosis["detection"] = {
            "machine_type":    resolved,
            "confidence":      detection.confidence,
            "source":          detection.source,
            "clip_confidence": detection.clip_confidence,
            "audio_confidence": detection.audio_confidence,
            "gemini_used":     detection.gemini_used,
        }

        logger.info(f"✅ /diagnose complete steps={len(diagnosis.get('solution', {}).get('steps', []))}")
        diagnosis = normalize_diagnosis_response(diagnosis, resolved)
        return JSONResponse(content=diagnosis)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ Diagnosis failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/diagnose/stream")
@limiter.limit("5/minute")
async def diagnose_stream(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    machine_type: str = Form(default=""),   # optional hint
    language: str = Form(default="en"),
    # ── Fast-path: pre-extracted frames from Flutter (v4.3+) ─────────────────
    # When Flutter sends frame0/frame1/frame2 as JPEG files, the backend skips
    # all video decode (no cv2, no disk write of the .mp4) and runs CLIP directly
    # on these three ~50 KB images. The video field is still accepted for
    # backward-compatibility but its bytes are discarded if frames are present.
    # Upload size drops from ~25 MB to ~350 KB → reduces Stage-0 latency by
    # ~40–60 s on Indian rural mobile connections.
    frame0: Optional[UploadFile] = File(default=None),
    frame1: Optional[UploadFile] = File(default=None),
    frame2: Optional[UploadFile] = File(default=None),
    _auth: None = Depends(verify_app_key),
):
    """
    Streaming diagnosis — Server-Sent Events.
    Stage event format is IDENTICAL to v4.0 — zero Flutter changes needed.

    Security additions (v4.2):
      • X-App-Key header required (403 if missing/wrong)
      • 5 requests/minute/IP rate limit (429 if exceeded)
      • Video ≤ 20 MB, audio ≤ 5 MB, formats validated before stream starts
      • Transcribed text sanitised for prompt injection inside event_stream()
      • All Gemini calls wrapped with 15 s timeout + per-IP hourly cap
    """
    ip = request.client.host if request.client else "unknown"
    logger.info(f"📡 /diagnose/stream hint={machine_type!r} lang={language} ip={ip}")
    request_id = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]

    try:
        audio_bytes = await audio.read()
        video_bytes = await video.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload read failed: {exc}")

    # ── Fast path: read pre-extracted frames (Flutter v4.3+) ─────────────────
    # Flutter sends frame0/frame1/frame2 as individual JPEG uploads.
    # If present, we skip video decode entirely. The video bytes are still read
    # above (the field is required for backward-compat) but are never written
    # to disk when frames are present.
    pre_frames: list[bytes] = []
    for _fup in (frame0, frame1, frame2):
        if _fup is not None:
            try:
                _fb = await _fup.read()
                if len(_fb) > 512:          # ignore empty/stub fields
                    pre_frames.append(_fb)
            except Exception:
                pass
    _using_pre_frames = len(pre_frames) > 0
    if _using_pre_frames:
        logger.info(f"⚡ Fast-path: {len(pre_frames)} pre-extracted frames received — skipping video decode")

    # ── Security: validate files BEFORE starting the stream ──────────────────
    validate_audio_upload(audio.filename or "rec.m4a", audio_bytes, ip=ip)
    if not _using_pre_frames and video.filename and len(video_bytes) > 1024:
        validate_video_upload(video.filename, video_bytes, ip=ip)

    async def event_stream() -> AsyncIterator[str]:
        # ── Token-budget architecture ─────────────────────────────────────────
        # Goal: exactly 2 Gemini calls per /diagnose/stream request.
        #
        #   Call 1 — transcribe_audio_with_gemini  (audio→text, unavoidable)
        #   Call 2 — generate_diagnosis_with_gemini (reason once, output JSON)
        #
        # detect_machine uses local CLIP zero-shot + audio keyword matching.
        # Its internal Gemini fallback fires only when CLIP < 0.55 AND no
        # client hint AND no machine keyword found in the transcription.
        # We eliminate that third call by:
        #   a) running CLIP and transcription concurrently (asyncio.gather) so
        #      both results are available before we decide on machine type;
        #   b) resolving machine type from transcription text when CLIP is
        #      uncertain — so detect_machine's Gemini path is never reached.
        #
        # RAG (Stage 2) is fully local — zero Gemini calls.
        # ─────────────────────────────────────────────────────────────────────

        audio_path = UPLOAD_DIR / f"{request_id}_audio_{_safe_filename(audio.filename, 'rec.m4a')}"
        video_path: Optional[Path] = None
        try:
            # ── Save files ────────────────────────────────────────────────────
            # Audio is always saved (needed for transcription Gemini call).
            # Video is only saved when Flutter did NOT send pre-extracted frames.
            # When pre_frames are present the video bytes are discarded — no disk
            # I/O, no cv2 decode, no OpenCV VideoCapture.
            audio_path.write_bytes(audio_bytes)
            if not _using_pre_frames and video.filename and len(video_bytes) > 1024:
                video_path = UPLOAD_DIR / f"{request_id}_video_{_safe_filename(video.filename, 'rec.mp4')}"
                video_path.write_bytes(video_bytes)

            # ── Stages 0 + 1: CLIP detection & transcription — CONCURRENT ────
            # CLIP reads video bytes (local, CPU-only).
            # Transcription calls Gemini on audio bytes.
            # They are completely independent so we run them in parallel,
            # cutting the combined wall-clock time roughly in half.
            yield _sse("stage_start", {"stage": 0, "label": "Identifying your machine from the video"})
            yield _sse("stage_start", {"stage": 1, "label": "Understanding your voice complaint"})

            # ── Transcription: full pipeline with audio pre-processing ──────────
            _stream_raw_store:  list[str]  = [""]
            _stream_meta_store: list[dict] = [{}]

            async def _transcribe() -> str:
                if can_call_gemini(ip):
                    result = await gemini_with_timeout(
                        transcribe_audio_full(audio_path),
                        fallback=None,
                        context="stream/transcription",
                    )
                    record_gemini_call(ip)
                    if result and isinstance(result, dict):
                        _stream_raw_store[0]  = result.get("raw_transcript", "")
                        _stream_meta_store[0] = result
                        return result.get("normalized_problem") or "farm machine problem"
                    return "farm machine problem"
                logger.warning(f"⚠️  Gemini budget exceeded [{ip}] — fallback transcription")
                return "farm machine problem"

            async def _detect_clip() -> Any:
                # ── Fast path: pre-extracted frames from Flutter ──────────────
                # Uses detect_machine_from_frames() which runs the IDENTICAL
                # CLIP → fusion → Gemini pipeline without any video file I/O.
                if _using_pre_frames:
                    return await detect_machine_from_frames(
                        frame_bytes_list=pre_frames,
                        transcription_text=None,
                    )
                # ── Standard path: full video decode ─────────────────────────
                # Run CLIP-only detection first (local, fast, no Gemini).
                # We pass transcription_text=None here so detect_machine uses
                # only CLIP + audio keywords from the video.
                return await detect_machine(
                    video_path=video_path,
                    transcription_text=None,
                )

            # Run both concurrently — Gemini call and CLIP inference overlap
            problem_text_raw, detection_clip = await asyncio.gather(
                _transcribe(), _detect_clip()
            )

            # ── Prompt injection guard ────────────────────────────────────────
            problem_text = check_prompt_injection(
                problem_text_raw or "", ip=ip, field="stream/transcription"
            )
            # Strip "unknown machine" hallucination from transcription normalizer
            problem_text = re.sub(
                r"^The\s+unknown\s+machine'?s?\s+",
                "The ",
                problem_text,
                flags=re.IGNORECASE,
            )
            problem_text = re.sub(
                r"^The\s+unknown\s+machines'?\s+",
                "The ",
                problem_text,
                flags=re.IGNORECASE,
            )
            # Also catch "on/in the unknown machine" and "unknown machine has/is"
            problem_text = re.sub(
                r"\s+(?:on|in)\s+the\s+unknown\s+machine",
                "",
                problem_text,
                flags=re.IGNORECASE,
            )
            problem_text = re.sub(
                r"^The\s+unknown\s+machine\s+(?:has|is)\s+",
                "The ",
                problem_text,
                flags=re.IGNORECASE,
            )

            # ── Machine-context re-validation (zero extra API calls normally) ──
            _clip_hint_s = detection_clip.machine_type or ""
            _smeta       = _stream_meta_store[0]
            if (
                _clip_hint_s
                and _smeta.get("gemini_used") == "false"
                and _stream_raw_store[0]
            ):
                from services.transcription_service import (
                    _detect_symptom_buckets, _validate_output, _light_clean_raw
                )
                _raw_s   = _stream_raw_store[0]
                _bkts_s  = _detect_symptom_buckets(_raw_s)
                _ok_s, _reason_s = _validate_output(
                    _raw_s, problem_text, _bkts_s, _clip_hint_s
                )
                if not _ok_s:
                    logger.warning(
                        f"⚠️  Stream post-CLIP validation failed ({_reason_s}) "
                        f"— re-normalising hint={_clip_hint_s!r}"
                    )
                    if can_call_gemini(ip):
                        from services.transcription_service import _extract_with_gemini
                        problem_text = await _extract_with_gemini(
                            _raw_s, _bkts_s, _clip_hint_s
                        )
                        record_gemini_call(ip)
                        problem_text = check_prompt_injection(
                            problem_text, ip=ip, field="stream/context_revalidation"
                        )

            # ── Resolve machine type (no extra Gemini call) ───────────────────
            # Priority: (1) high-confidence CLIP, (2) client hint, (3) audio
            # keyword match in transcription.  detect_machine's own Gemini
            # fallback is only reached when all three signals are ambiguous —
            # and we never call it here because we re-run detect_machine with
            # the transcription text if CLIP is uncertain, which gives the
            # keyword matcher enough signal to avoid the Gemini path.
            resolved = detection_clip.machine_type
            detection = detection_clip

            if detection_clip.confidence < 0.55:
                # CLIP is uncertain — run a second pass WITH the transcription text
                # so the audio keyword matcher can boost or correct the result.
                #
                # CRITICAL DIFFERENCE from the standard path:
                # When frames were pre-extracted (fast path), we must NOT call
                # detect_machine(video_path=None) because that builds a new
                # DetectionResult with zero CLIP signal and overwrites the machine
                # type with the weak audio-only result (0.155 in the logs).
                # Instead we re-run detect_machine_from_frames with the same frames
                # AND the transcription text — CLIP + audio fusion, not audio alone.
                if _using_pre_frames:
                    detection = await detect_machine_from_frames(
                        frame_bytes_list=pre_frames,
                        transcription_text=problem_text,
                    )
                    if detection.gemini_used:
                        record_gemini_call(ip)
                else:
                    # Standard path: re-run with transcription text so audio keyword
                    # matcher resolves without Gemini. CRITICAL: video_path=None so
                    # the video is NOT re-decoded a second time.
                    detection = await detect_machine(
                        video_path=None,
                        transcription_text=problem_text,
                    )
                    # Restore the frames from the first (and only) CLIP decode
                    detection.frames = detection_clip.frames
                    if detection.gemini_used:
                        record_gemini_call(ip)
                resolved = detection.machine_type

            # Client hint overrides when our detection is still uncertain
            if machine_type.strip():
                hint = resolve_machine_id(machine_type.strip())
                if detection.confidence < 0.55 and get_profile(hint):
                    logger.info(f"🔧 Using client hint {hint!r} (conf={detection.confidence:.2f})")
                    resolved = hint

            _smeta_final = _stream_meta_store[0]
            yield _sse("stage_done", {
                "stage":            1,
                "label":            "Understanding your voice complaint",
                "transcription":    problem_text,
                "raw_transcript":   _stream_raw_store[0] or problem_text,
                "transcription_meta": {
                    "quality_grade": _smeta_final.get("quality_grade", ""),
                    "gemini_used":   _smeta_final.get("gemini_used", ""),
                    "skip_reason":   _smeta_final.get("skip_reason", ""),
                },
            })
            yield _sse("stage_done", {
                "stage": 0, "label": "Identifying your machine from the video",
                "machine_type": resolved,
                "detection_confidence": detection.confidence,
                "detection_source":     detection.source,
            })

            # ── Stage 2: Pipeline (Phase 1–3) — RAG + DB lock + visual gate ────
            # Reuse frames already decoded by CLIP — zero extra video I/O.
            # detection.frames holds the early/mid/late JPEG bytes that CLIP
            # quality-scored during Stage 0. No re-open, no re-decode, no cv2.
            clip_frames  = detection.frames
            mid_frame    = clip_frames[len(clip_frames) // 2] if clip_frames else None
            visual_hash  = _phash_bytes(mid_frame) if mid_frame else ""
            logger.info(f"📸 Frames from CLIP cache: {len(clip_frames)}/3  phash={visual_hash or 'none'}")

            yield _sse("stage_start", {"stage": 2, "label": "Searching repair manuals for your issue"})

            # Phase 1 router is skipped — machine already resolved via CLIP.
            # Use machine override only when CLIP confidence is high
            # When low, let the router determine machine type from query keywords
            _machine_override = resolved if detection.confidence >= 0.70 else None
            pipeline = await run_full_pipeline(
                query=problem_text,
                vector_db=vector_db,
                frame_bytes=mid_frame,
                language=language,
                machine_type_override=_machine_override,
            )

            if pipeline.blocked:
                logger.info(f"🔒 /diagnose/stream pipeline blocked: {pipeline.block_reason}")
                block_response = pipeline.response
                block_response["request_id"]    = request_id
                block_response["transcription"] = problem_text
                block_response["detection"]     = {
                    "machine_type":     resolved,
                    "confidence":       detection.confidence,
                    "source":           detection.source,
                    "clip_confidence":  detection.clip_confidence,
                    "audio_confidence": detection.audio_confidence,
                    "gemini_used":      detection.gemini_used,
                }
                yield _sse("stage_done", {
                    "stage": 2, "label": "Searching repair manuals for your issue",
                    "rag_chunks": 0, "rag_active": False,
                    "pipeline_blocked": True,
                    "block_reason": pipeline.block_reason,
                })
            
                yield _sse("stage_done", {
                    "stage": 3, "label": "Preparing your step-by-step repair guide",
                    "result": block_response,
                    "cache_hit": False,
                })
                yield _sse("done", {})
                return

            rag_context = pipeline.rag_context
            rag_chunks  = len(rag_context.split("---")) if rag_context else 0

            yield _sse("stage_done", {
                "stage": 2, "label": "Searching repair manuals for your issue",
                "rag_chunks": rag_chunks, "rag_active": rag_chunks > 0,
                "pipeline_blocked": False,
                "rag_score": round(pipeline.lock.score, 3) if pipeline.lock else 0,
            })

            # ── Stage 3: Diagnosis — ONE Gemini call, multimodal 3-frame ─────
            # Cache key includes visual_hash of mid frame: same machine + same
            # symptom cluster but different visual state → different cache entry.
            # "water flowing" and "pump completely dead" get separate cached plans.
            yield _sse("stage_start", {"stage": 3, "label": "Preparing your step-by-step repair guide"})
            knowledge = load_knowledge_base(pipeline.machine_type)

            plan_cache_key = _plan_cache_key(pipeline.machine_type, problem_text, visual_hash)
            cached_plan    = _plan_cache_get(plan_cache_key)

            if cached_plan is not None:
                diagnosis = cached_plan
                diagnosis["cache_hit"] = True
                logger.info(f"🎯 stream cache HIT key={plan_cache_key}")
                diagnosis = normalize_diagnosis_response(diagnosis, pipeline.machine_type)
            elif can_call_gemini(ip):
                diagnosis = await gemini_with_timeout(
                    generate_diagnosis_with_gemini(
                        machine_type=pipeline.machine_type,
                        problem_text=problem_text,
                        language=language,
                        rag_context=rag_context,
                        knowledge_base=knowledge,
                        visual_frames=clip_frames,
                        top_score=round(pipeline.lock.score, 3) if pipeline.lock else 0.0,
                    ),
                    fallback=None,
                    context="stream/diagnosis",
                )
                if diagnosis:
                    record_gemini_call(ip)
                    # Enrich with visual gate findings before caching
                    if pipeline.gate:
                        diagnosis["confirmed_part"]     = pipeline.gate.part_id
                        diagnosis["visual_observation"] = pipeline.gate.fault_description
                        diagnosis["rag_score"]          = round(pipeline.lock.score, 3)
                    _plan_cache_set(plan_cache_key, diagnosis)
                else:
                    diagnosis = _local_fallback_diagnosis(resolved, problem_text)
            else:
                logger.warning(f"⚠️  Gemini budget exceeded [{ip}] — fallback diagnosis")
                diagnosis = _local_fallback_diagnosis(resolved, problem_text)

            diagnosis["request_id"]      = request_id
            diagnosis["transcription"]    = problem_text
            diagnosis["raw_transcript"]   = _stream_raw_store[0] or problem_text
            _smeta_result = _stream_meta_store[0]
            diagnosis["transcription_meta"] = {
                "quality_grade": _smeta_result.get("quality_grade", ""),
                "gemini_used":   _smeta_result.get("gemini_used", ""),
                "skip_reason":   _smeta_result.get("skip_reason", ""),
            }
            diagnosis["detection"] = {
                "machine_type":    resolved,
                "confidence":      detection.confidence,
                "source":          detection.source,
                "clip_confidence": detection.clip_confidence,
                "audio_confidence": detection.audio_confidence,
                "gemini_used":     detection.gemini_used,
            }
            diagnosis = normalize_diagnosis_response(diagnosis, pipeline.machine_type)
            yield _sse("stage_done", {
                "stage": 3, "label": "Preparing your step-by-step repair guide",
                "result": diagnosis,
                "cache_hit": diagnosis.get("cache_hit", False),
            })

            yield _sse("done", {})
            logger.info(f"✅ /diagnose/stream complete [{request_id}]")

        except Exception as exc:
            logger.error(f"❌ Stream error [{request_id}]: {exc}")
            yield _sse("error", {"message": str(exc)})
        finally:
            # Windows locks files held by sockets that closed uncleanly (WinError 32).
            # Retry up to 3 times with a short delay before giving up — the OS
            # releases the handle within ~200ms of the SSL connection teardown.
            async def _safe_unlink(p: Path) -> None:
                for attempt in range(3):
                    try:
                        p.unlink(missing_ok=True)
                        return
                    except PermissionError:
                        if attempt < 2:
                            await asyncio.sleep(0.3)
                        else:
                            logger.warning(f"⚠️  Could not delete temp file (still locked): {p.name}")

            await _safe_unlink(audio_path)
            if video_path:
                await _safe_unlink(video_path)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )



# ─────────────────────────────────────────────────────────────────────────────
# /locate_part  — AR guidance loop
#
# Called by Flutter every 3s while user is in locating state.
# Returns normalised bbox [cx,cy,w,h] + camera guidance string.
# Never returns an arrow location if confidence < 0.72 or any reject flag set.
# Token cost: ~573 tokens per call (~$0.00007)
# ─────────────────────────────────────────────────────────────────────────────
from services.locate_part_service import locate_part_with_gemini   # noqa: E402
from services.safety_guards import check_multiple_texts
@app.post("/locate_part")
@limiter.limit("30/minute")
async def locate_part(
    request: Request,
    image: UploadFile = File(...),
    required_part: str  = Form(...),
    area_hint:     str  = Form(default=""),
    machine_type:  str  = Form(default="tractor"),
    attempt_count: int  = Form(default=1),
    language:      str  = Form(default="en"),
    frame_id:      int  = Form(default=0),
    search_roi:    str  = Form(default=''),  # 'cx,cy,margin' or '' if no prior bbox
    _auth: None = Depends(verify_app_key),
):
    """
    AR part-location endpoint.

    Flutter sends a single JPEG frame captured from the live camera preview.
    Returns either:
      { found: true,  bbox: [cx,cy,w,h], confidence: 0.85, camera_guidance: "..." }
      { found: false, bbox: null,         confidence: 0.0,  camera_guidance: "Move to pump_body" }

    Security: same guards as verify_step (rate-limit, app-key, length caps).
    """
    ip = request.client.host if request.client else "unknown"

    # Input length caps
    if len(required_part) > 120:
        required_part = required_part[:120]
    if len(area_hint) > 120:
        area_hint = area_hint[:120]

    # Prompt injection guard on user-supplied fields
    required_part = check_prompt_injection(required_part, ip=ip, field="required_part")
    area_hint     = check_prompt_injection(area_hint,     ip=ip, field="area_hint")

    logger.info(
        f"🎯 /locate_part part={required_part} area={area_hint} "
        f"attempt={attempt_count} ip={ip}"
    )

    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image")
        if len(image_bytes) > 5_000_000:   # 5 MB hard cap
            raise HTTPException(status_code=413, detail="Image too large")

        if can_call_gemini(ip):
            # Parse search_roi: 'cx,cy,margin' → (float, float, float) | None
            _parsed_roi = None
            if search_roi:
                try:
                    _parts = [float(x) for x in search_roi.split(',')]
                    if len(_parts) == 3:
                        _parsed_roi = tuple(_parts)
                except (ValueError, TypeError):
                    pass

            result = await gemini_with_timeout(
                locate_part_with_gemini(
                    image_bytes    = image_bytes,
                    required_part  = required_part,
                    area_hint      = area_hint,
                    machine_type   = machine_type,
                    attempt_count  = attempt_count,
                    language       = language,
                    frame_id       = frame_id,
                    search_roi     = _parsed_roi,
                ),
                fallback = None,
                context  = "locate_part",
            )
            if result:
                record_gemini_call(ip)
            else:
                result = {
                    "found": False, "bbox": None, "confidence": 0.0,
                    "camera_guidance": "Analysis timed out — please try again.",
                    "part_description": None, "reject_flags": {},
                }
        else:
            logger.warning(f"⚠️  Gemini budget exceeded [{ip}] — locate_part skipped")
            result = {
                "found": False, "bbox": None, "confidence": 0.0,
                "camera_guidance": "Service busy — try again in a moment.",
                "part_description": None, "reject_flags": {},
            }

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ /locate_part error [{ip}]: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "found": False, "bbox": None, "confidence": 0.0,
                "camera_guidance": "Could not analyse frame — hold still and retry.",
                "part_description": None, "reject_flags": {},
            },
        )
    
from services.safety_guards import check_multiple_texts

@app.post("/verify_step")
@limiter.limit("20/minute")
async def verify_step(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    step_text: str = Form(...),
    required_part: str = Form(default="machine_part"),
    area_hint: str = Form(default="engine_compartment"),
    machine_type: str = Form(default="tractor"),
    problem_context: str = Form(default=""),
    attempt_count: int = Form(default=1),
    previous_steps: str = Form(default="[]"),
    language: str = Form(default="en"),
    include_hindi: str = Form(default="false"),
    is_blind_search: bool = Form(default=False),
    _auth: None = Depends(verify_app_key),
):
    """Step verification endpoint — unchanged interface, delegates to verification_service.

    Security additions (v4.2):
      • X-App-Key header required
      • 20 requests/minute/IP rate limit
      • step_text sanitised for prompt injection
      • Gemini call wrapped with 15 s timeout + per-IP hourly cap
    """
    ip = request.client.host if request.client else "unknown"

    # ── Security: input length caps ───────────────────────────────────────────
    # Prevents oversized payloads bloating Gemini prompts / slowing the server.
    # step_text raises 400 (Flutter must send valid input); others are silently
    # truncated since they're contextual and partial content is still useful.
    if len(step_text) > 1000:
        raise HTTPException(status_code=400, detail="step_text exceeds 1000 character limit")
    if len(problem_context) > 500:
        problem_context = problem_context[:500]
    if len(previous_steps) > 8000:
        previous_steps = "[]"   # silently drop oversized history — non-critical

        # ── Security: previous_steps — parse, validate, cap, sanitize ────────────
    # F5 fix: previous_steps is embedded directly into the Gemini prompt as a
    # history block. We must sanitize every string recursively — attackers can
    # hide injection payloads in any field, not just the ones we know today.
    _ps_parsed: list = []
    MAX_HISTORY = 20

    try:
        _ps_parsed = json.loads(previous_steps)
    except (json.JSONDecodeError, ValueError):
        _ps_parsed = []

    # Ensure it's a list of dicts only
    if isinstance(_ps_parsed, list):
        _ps_parsed = [x for x in _ps_parsed if isinstance(x, dict)]
    else:
        _ps_parsed = []

    # Cap history to prevent context-window abuse
    if len(_ps_parsed) > MAX_HISTORY:
        _ps_parsed = _ps_parsed[-MAX_HISTORY:]

    # Recursive sanitization — catches injection in ANY field, now and future
    if _ps_parsed:
        _ps_parsed = sanitize_prompt_object(_ps_parsed, ip=ip, field="previous_steps")

    # Re-serialize with ensure_ascii=False so Hindi stays in Devanagari
    previous_steps = json.dumps(_ps_parsed, ensure_ascii=False)

    # If parsing gave us nothing, derive part/area from step_text
    if not _ps_parsed:
        required_part, area_hint = derive_part_and_area(step_text, machine_type)

    # ── Security: prompt injection on step_text + problem_context ─────────────
    step_text       = check_prompt_injection(step_text,        ip=ip, field="step_text")
    problem_context = check_prompt_injection(problem_context,  ip=ip, field="problem_context")

    logger.info(f"👁️ verify_step attempt={attempt_count} part={required_part} area={area_hint} ip={ip}")

    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image")

        # Gemini vision call — guarded
        if can_call_gemini(ip):
            if is_blind_search:
                # ── PHASE 1: BLIND SEARCH PROMPT ──
                prompt = f"""You are guiding a farmer to find {required_part} on a {machine_type}.
The farmer is pointing their camera at SOMETHING on the machine, but you need to identify what they're looking at.

1. Identify what component is visible in this image.
2. If it's the {required_part}: return verified=true with bbox.
3. If it's a DIFFERENT component: return verified=false, tell them what they found, 
   and give a guidance_vector (left/right/up/down) toward the {required_part}.
   Also describe WHERE the {required_part} is relative to what they're seeing.

Return ONLY JSON:
{{"verified": true/false, "detected_part": "what you see", 
  "guidance_vector": "left/right/up/down or empty", 
  "feedback_en": "guidance", "feedback_hi": "Hindi guidance",
  "bbox": [ymin,xmin,ymax,xmax]}}"""
                
                # Use the new resilient vision_call helper!
                from utils.vision_client import vision_call
                from utils.json_repair import repair_json
                try:
                    raw_resp = await vision_call(prompt, image_bytes)
                    result = repair_json(raw_resp)
                except Exception as e:
                    logger.error(f"Blind search vision_call error: {e}")
                    result = None
            else:
                # ── STANDARD VERIFY STEP ──
                result = await gemini_with_timeout(
                    verify_step_with_gemini(
                        image_bytes=image_bytes,
                        step_text=step_text,
                        required_part=required_part,
                        area_hint=area_hint,
                        machine_type=machine_type,
                        problem_context=problem_context,
                        attempt_count=attempt_count,
                        language=language,
                        include_hindi=(include_hindi.lower() == "true"),
                        previous_steps=previous_steps,  # ← semantic memory wiring
                    ),
                    fallback=None,
                    context="verify_step",
                )
                
            if result:
                record_gemini_call(ip)
            else:
                # Timeout fallback
                result = {
                    "status": "unclear", "verified": False, "confidence": 0.0,
                    "detected_part": "Analysis timed out",
                    "correct_part": required_part, "machine_type": machine_type,
                    "ai_observation": "The analysis took too long. Please try again.",
                    "feedback": "Move closer and hold still — tap Analyze again.",
                    "feedback_hi": "कैमरे को करीब लाएं और स्थिर रखें — फिर विश्लेषण दबाएं।",
                    "attempt_count": attempt_count,
                }
        else:
            logger.warning(f"⚠️  Gemini budget exceeded [{ip}] — skipping verify_step Gemini call")
            result = {
                "status": "unclear", "verified": False, "confidence": 0.0,
                "detected_part": "Service temporarily limited",
                "correct_part": required_part, "machine_type": machine_type,
                "ai_observation": "Too many requests. Please wait a moment and try again.",
                "feedback": "Please try again in a few minutes.",
                "feedback_hi": "कुछ मिनट बाद फिर से कोशिश करें।",
                "attempt_count": attempt_count,
            }

        result["request_id"] = hashlib.md5(f"{datetime.now()}{attempt_count}".encode()).hexdigest()[:8]
        logger.info(f"✅ verify_step={result.get('status')} conf={result.get('confidence', 0):.2f}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ Verification failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/inspect_part")
@limiter.limit("30/minute")
async def inspect_part(
    image: UploadFile = File(...),
    machine_type: str = Form(...),
    required_part: str = Form(...),
    area_hint: str = Form("engine_compartment"),
    language: str = Form("en"),
    request: Request = None,
):
    """
    INSPECTION endpoint — analyzes a frozen frame for damage AFTER verification.
    Returns: outcome, severity, damage_region, observations, repairability.
    Does NOT verify part identity — that's /verify_step's job.
    """
    # Auth + rate limit checks...
    
    image_bytes = await image.read()
    
    result = await inspect_part_service(
        image_bytes=image_bytes,
        machine_type=machine_type,
        required_part=required_part,
        area_hint=area_hint,
        language=language,
    )
    
    return result

@app.post("/verify_fix")
@limiter.limit("20/minute")
async def verify_fix(
    request: Request,
    image: UploadFile = File(...),
    instruction: str = Form(...),
    expected_result: str = Form(...),
    machine_type: str = Form(default="water_pump"),
    language: str = Form(default="en"),
    _auth: None = Depends(verify_app_key),
):
    ip = request.client.host if request.client else "unknown"
    
    if len(instruction) > 500:
        instruction = instruction[:500]
    if len(expected_result) > 500:
        expected_result = expected_result[:500]
    
    instruction = check_prompt_injection(instruction, ip=ip, field="instruction")
    expected_result = check_prompt_injection(expected_result, ip=ip, field="expected_result")
    
    logger.info(f"🔧 /verify_fix machine={machine_type} instruction='{instruction[:60]}...'")
    
    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image")
        if len(image_bytes) > 5_000_000:
            raise HTTPException(status_code=413, detail="Image too large")
        
        prompt = f"""You are an expert agricultural mechanic inspector for Indian farmers.
You are inspecting whether a repair action was completed correctly.

MACHINE: {machine_type}
INSTRUCTION GIVEN TO FARMER: "{instruction}"
EXPECTED RESULT IF DONE CORRECTLY: "{expected_result}"

Look at this image carefully. Has the farmer completed the repair correctly?
Check for: correct part used, proper installation, remaining damage (rust/cracks/burns), 
loose connections, safety concerns.

Be strict but fair. If something small is wrong, tell them exactly what to fix.

Return ONLY this JSON (no markdown):
{{"verified": true/false, "repair_quality": "good/partial/failed", 
  "feedback_en": "specific guidance in simple English", 
  "feedback_hi": "same guidance in simple Hindi for rural farmer",
  "issues_found": ["list of remaining issues, empty if verified"]}}"""
        
        # Use the vision client with automatic failover
        from utils.vision_client import vision_call
        from utils.json_repair import repair_json
        
        raw_response = await vision_call(prompt, image_bytes, max_tokens=400, temperature=0.1)
        result = repair_json(raw_response)
        
        result["request_id"] = hashlib.md5(
            f"{datetime.now()}{instruction}".encode()
        ).hexdigest()[:8]
        
        logger.info(f"✅ /verify_fix verified={result.get('verified')} quality={result.get('repair_quality')}")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ /verify_fix failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "verified": False,
                "repair_quality": "error",
                "feedback_en": "Could not analyze the image. Please try again.",
                "feedback_hi": "छवि का विश्लेषण नहीं कर सका। कृपया पुनः प्रयास करें।",
                "issues_found": [str(exc)[:100]]
            }
        )

@app.post("/safety_check")
async def safety_check(image: UploadFile = File(...)):
    """Continuous safety monitoring endpoint."""
    logger.info("🛡️ Safety check request")
    try:
        image_bytes = await image.read()
        result = await safety_check_with_gemini(image_bytes)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("❌ Safety endpoint failed — failing closed")
        return JSONResponse(
            status_code=503,
            content={
                "safe": False,
                "analysis_available": False,
                "hazard_detected": "Safety analysis unavailable.",
                "severity": "high",
                "warning_message": "STOP. Safety verification could not be completed.",
                "error_code": "SAFETY_SERVICE_UNAVAILABLE",
            },
        )

@app.delete("/plan_cache")
async def clear_plan_cache(
    request: Request,
    _auth: None = Depends(verify_app_key),
):
    """
    Clear all cached repair plans (operator tool).
    Use after major knowledge base updates to force fresh LLM generation.
    """
    files = list(PLAN_CACHE_DIR.glob("*.json"))
    for f in files:
        f.unlink(missing_ok=True)
    logger.info(f"🗑️  Plan cache cleared by {request.client.host}: {len(files)} entries removed")
    return {"status": "cleared", "entries_removed": len(files)}


@app.delete("/plan_cache/{machine_type}")
async def clear_plan_cache_for_machine(
    machine_type: str,
    _auth: None = Depends(verify_app_key),
):
    """
    Selectively invalidate all cached plans for a specific machine type.
    Useful when you update the knowledge base for one machine only.
    """
    # We can't reverse the SHA-256 but we can scan and match by checking
    # a small metadata sidecar. For simplicity, clear all — this is a rare op.
    files = list(PLAN_CACHE_DIR.glob("*.json"))
    removed = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("solution", {}).get("machine_type", "") == machine_type:
                f.unlink(missing_ok=True)
                removed += 1
        except Exception:
            pass
    logger.info(f"🗑️  Plan cache: {removed} entries removed for machine={machine_type}")
    return {"status": "cleared", "machine_type": machine_type, "entries_removed": removed}


@app.post("/feedback")
async def submit_feedback(
    rating: int = Form(...),
    comments: str = Form(default=""),
    step_index: int = Form(default=0),
    machine_type: str = Form(default="unknown"),
):
    logger.info(f"📝 Feedback: rating={rating} machine={machine_type}")
    # Sanitise: strip newlines (prevent log-injection forgery) and cap length
    safe_comments    = comments.replace("\n", " ").replace("\r", " ").replace(",", ";")[:500]
    safe_machine     = machine_type.replace("\n", "").replace("\r", "")[:50]
    feedback_file = Path("user_feedback.log")
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()},{safe_machine},{step_index},{rating},{safe_comments}\n")
    return {"status": "thank you"}


# ============================================================================
# API ENDPOINTS — NEW AGENT SYSTEM
# ============================================================================

@app.post("/agent/session", response_model=CreateSessionResponse)
async def create_agent_session(request: CreateSessionRequest):
    """
    Create a new stateful repair session for ANY supported farm machine.
    If machine_type is not provided (or empty), Phase 1 router auto-detects
    it from the problem_description using a lightweight LLM call.
    """
    # ── Phase 1: Dynamic machine-type resolution ──────────────────────────────
    if request.machine_type and request.machine_type.strip():
        # Caller supplied a machine_type — normalise alias → canonical ID
        canonical_type = resolve_machine_id(request.machine_type)
        if not canonical_type:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown machine type: '{request.machine_type}'. Call GET /machines for the full list.",
            )
    else:
        # No machine_type provided — use Phase 1 router to extract from problem description
        router_result = await resolve_machine_from_query(request.problem_description)
        logger.info(
            f"🔍 Router auto-detected machine: {router_result.machine_type} "
            f"(conf={router_result.confidence:.2f}) from: '{request.problem_description[:60]}'"
        )
        if router_result.machine_type == "unknown" or not router_result.router_ok:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not identify the machine type from your description. "
                    "Please mention the machine name (e.g. 'water pump', 'tractor', 'harvester') "
                    "or pass machine_type explicitly."
                ),
            )
        canonical_type = router_result.machine_type

    profile = get_profile(canonical_type)
    machine_label = profile.label_en if profile else canonical_type

    session = session_manager.create_session(
        machine_type=canonical_type,
        problem=request.problem_description,
        language=request.language,
        diagnosis_steps=request.diagnosis_steps,
    )
    logger.info(f"🆕 Agent session created: {session.session_id} machine={canonical_type}")
    return CreateSessionResponse(
        session_id=session.session_id,
        message=f"Repair session started for {machine_label}. Call /agent/next to begin diagnosis.",
    )


@app.post("/agent/next")
async def agent_next(request: AgentNextRequest):
    """Creates a new stateful repair session for farm machinery."""
    session = session_manager.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found or expired. Create a new session via /agent/session.",
        )

    logger.info(
        f"🤖 /agent/next session={request.session_id} "
        f"stage={session.current_stage} "
        f"parts_verified={len(session.verified_parts)}"
    )

    try:
        response = await repair_agent.decide_next_step(
            session=session,
            last_verification=request.last_verification_result,
        )
        # Immediately after creating RepairSession
        logger.info("===== DEBUG: Repair Session Initialized =====")
        logger.info("Session ID: %r", session.session_id)
        logger.info("Initial current_step_id: %r", session.current_step_id)
        logger.info("Repair plan steps:")
        for i, step in enumerate(session.repair_plan.steps):
            logger.info("  %d: id=%r, action=%s", i, step.step_id, step.action)
        logger.info("===========================================")
        # Persist updated session
        session_manager.update_session(session)

        # If resolved or escalated, clean up session after responding
        if response.status in ("resolved", "escalate", "unsafe"):
            logger.info(f"🏁 Session {request.session_id} terminal status: {response.status}")

        return JSONResponse(content=response.model_dump())

    except InvalidRepairPlan:
        raise  # let the exception handler above return 503
    except Exception as exc:
        logger.error(f"❌ Agent reasoning failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")


@app.get("/agent/session/{session_id}")
async def get_agent_session(
    session_id: str,
    _auth: None = Depends(verify_app_key),
):
    """
    Inspect the current state of an active repair session.
    Gated behind X-App-Key — exposes verified_parts, observations, and
    problem description which must not be publicly readable.
    """
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or expired.")
    return JSONResponse(content=session.model_dump())


@app.delete("/agent/session/{session_id}")
async def delete_agent_session(
    session_id: str,
    _auth: None = Depends(verify_app_key),
):
    """Explicitly end and clean up a repair session."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"status": "deleted", "session_id": session_id}

from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# EVAL ENDPOINT  v2.0 — evaluate_text_rag
# ─────────────────────────────────────────────────────────────────────────────
# BUG FIX: Previously EvalRequest defaulted machine_type="tractor" and the
# evaluate_rag.py harness never sent machine_type in the payload. This meant
# ALL 8 evaluation queries (water pump, rotavator, harvester, etc.) were
# diagnosed as tractor problems — causing catastrophic wrong-machine
# hallucinations (TC_018: water pump → tractor cooling system diagnosis).
#
# FIX:
#   1. EvalRequest.machine_type now has NO default — the caller MUST send it.
#   2. If not sent (legacy clients), we infer machine_type from the query text
#      via a keyword map rather than blindly defaulting to "tractor".
#   3. evaluate_rag.py v2.0 now always sends machine_type inferred from
#      eval_dataset.json items or the _infer_machine_type() helper.

# ─────────────────────────────────────────────────────────────────────────────
# EVAL ENDPOINT  v2.0 — evaluate_text_rag
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_MACHINE_KEYWORDS = {
    "submersible_pump": ["submersible", "borewell", "deep well", "tube well"],
    "water_pump":       ["centrifugal pump", "water pump", "suction pipe", "foot valve",
                         "volute", "priming", "centrifugal"],
    "electric_motor":   ["electric motor", "motor winding", "capacitor", "relay", "mcb"],
    "harvester":        ["harvester", "combine", "header", "threshing drum", "chaff"],
    "thresher":         ["thresher", "threshing", "feed inlet", "concave"],
    "rotavator":        ["rotavator", "rotary tiller", "tine", "shear bolt", "pto"],
    "chaff_cutter":     ["chaff cutter", "fodder cutter"],
    "power_tiller":     ["power tiller", "walk-behind"],
    "diesel_engine":    ["diesel engine", "stationary engine"],
    "generator":        ["generator", "genset", "avr", "alternator"],
    "sprayer":          ["sprayer", "spray pump", "nozzle", "chemical spray", "power sprayer"], # FIXED NAME
    "drone":            ["drone", "uav", "agri drone"],
    "tractor":          ["tractor"],
}

def _infer_machine_type_from_query(query: str) -> str:
    """Keyword-score the query using regex word boundaries.

    BUG 10 FIX: When the local keyword table has no match, delegate to
    route_query() (which uses KNOWN_MACHINE_IDS / _MACHINE_ALIAS_MAP from
    query_router.py) instead of blindly returning "tractor".
    """
    q = query.lower()
    scores: dict = {}
    for machine, keywords in _EVAL_MACHINE_KEYWORDS.items():
        score = sum(1 for kw in keywords if re.search(rf'\b{re.escape(kw)}\b', q))
        if score:
            scores[machine] = score
    if scores:
        return max(scores, key=lambda m: scores[m])

    # BUG 10 FIX: fall back to the canonical router instead of hard-coding "tractor"
    try:
        from query_router import route_query
        routed = route_query(query)
        if routed and routed.machine_type and routed.machine_type != "unknown":
            return routed.machine_type
    except Exception:
        pass  # query_router unavailable — degrade gracefully

    logger.warning("_infer_machine_type_from_query: no match for query='%s', defaulting to unknown", query[:60])
    return "unknown"

class EvalRequest(BaseModel):
    query: str
    machine_type: str = ""   # empty = will be inferred from query text

@app.post("/evaluate_text_rag")
async def evaluate_text_rag(req: EvalRequest):
    """
    Evaluation endpoint for RAG accuracy testing without audio/video.
    Runs the full Phase 1–3 pipeline (Phase 3 visual gate is auto-skipped
    when no frame is provided — correct behaviour for text-only eval).
    """
    logger.info(f"📊 Eval: machine={req.machine_type!r} | query={req.query[:60]}")

    from rag import _detect_language          # add at top of file
    pipeline = await run_full_pipeline(
        query=req.query,
        vector_db=vector_db,
        frame_bytes=None,
        language=_detect_language(req.query),  # ← auto-detect Hindi/English
        machine_type_override=req.machine_type.strip() or None,
    )

    if pipeline.blocked:
        # FIX 3: Handle OOD guard specifically with category logging
        if pipeline.phase_reached == "ood_guard":
            logger.info("📊 Eval blocked: ood_guard (category=%s)",
                        pipeline.response.get("ood_category", "unknown"))
        else:
            logger.info(f"📊 Eval blocked: {pipeline.block_reason}")
        response = pipeline.response
        # P0 FIX: Include rag_score in blocked eval responses so test suite can read it
        if pipeline.lock and pipeline.lock.score is not None:
            response["rag_score"] = round(pipeline.lock.score, 3)
        return JSONResponse(content=response)

    resolved_machine = pipeline.machine_type
    rag_score = round(pipeline.lock.score, 3) if pipeline.lock else 0.0
    logger.info(f"📊 Eval pipeline passed: machine={resolved_machine} rag_score={rag_score:.3f}")

    knowledge = load_knowledge_base(resolved_machine)
    diagnosis = await generate_diagnosis_with_gemini(
        machine_type=resolved_machine,
        problem_text=req.query,
        language="en",
        rag_context=pipeline.rag_context,
        knowledge_base=knowledge,
        visual_frames=[],
        top_score=round(pipeline.lock.score, 3) if pipeline.lock else 0.0,
    )

    # MIGRATED: Groq format normalization — ensure top-level lowercase status
    from services.diagnosis_service import _normalize_status  # MIGRATED: Groq format normalization
    diagnosis = _normalize_status(diagnosis)  # MIGRATED: Groq format normalization

    diagnosis = normalize_diagnosis_response(diagnosis, resolved_machine)
    diagnosis["rag_score"] = rag_score
    return JSONResponse(content=diagnosis)
# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 80)
    logger.info("AgriFix v4.2 Production Backend Starting...")
    logger.info("=" * 80)
    logger.info("Gemini API: ✅ Configured")
    logger.info("Machine Detection: MobileCLIP-S1 + audio keywords + Gemini fallback")
    logger.info("Agent: Stateful repair agent (Gemini + safety rules)")
    logger.info("Security: rate limiting + Gemini guard + file validation + auth")
    logger.info(
        f"📐 Limits enforced: video≤{VIDEO_MAX_BYTES//1_048_576}MB/{int(VIDEO_MAX_SECONDS)}s  "
        f"audio≤{AUDIO_MAX_BYTES//1_048_576}MB/{int(AUDIO_MAX_SECONDS)}s  "
        f"gemini≤{GEMINI_HOURLY_LIMIT}/IP/hr"
    )
    logger.info(f"Cache: {CACHE_DIR}")
    logger.info(f"Knowledge Base: {KB_DIR}")
    logger.info("=" * 80)

    port = int(os.environ.get("PORT", 7860))

    # Production stability settings — DO NOT change these without reading below:
    #
    # workers=1:
    #   SlowAPI rate limiter and Gemini credit guard use in-memory dicts.
    #   Multiple workers = separate memory = each worker gets the full quota.
    #   On HuggingFace free tier, 1 worker is correct. If you scale to multi-
    #   worker in future, replace the dicts with Redis.
    #
    # limit_concurrency=50:
    #   Caps simultaneous open requests. Each /diagnose/stream holds a connection
    #   for 5-15 s while Gemini runs. 50 concurrent = HF free tier RAM safe.
    #   Excess requests get HTTP 503 rather than OOM-killing the server.
    #
    # timeout_keep_alive=10:
    #   Disconnects idle keep-alive connections after 10 s. Prevents connection
    #   exhaustion from clients that open but don't close properly (mobile apps
    #   on bad networks). Standard production setting.

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENV") == "development",
        workers=1,
        limit_concurrency=50,
        timeout_keep_alive=10,
    )