"""
extract_manuals.py — AgriFix Hybrid PDF Pipeline
=================================================
PRIMARY:  PyMuPDF4LLM  — fast, low-RAM, native text layer extraction
FALLBACK: Docling       — triggered only when primary quality is poor,
                          PDF is scanned, tables are heavy, or text
                          layer is missing.

Decision flow
─────────────
  Digital PDF?
      ↓
  PyMuPDF4LLM
      ↓
  Good extraction? (heuristic checks)
      ↓
  YES → continue with chunks
  NO  → Fallback to Docling for that PDF
"""

import os
import re
import json
import time
import logging
import unicodedata
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
from utils.json_repair import repair_json

from groq import Groq
load_dotenv()

from google import genai
from pydantic import BaseModel, Field
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ── BGE-M3 embedding model ────────────────────────────────────────────────────
# pip install sentence-transformers numpy
# CrossEncoder (reranker) is NOT imported here — it lives in crossencoder_reranker.py
# and is only needed at query time, not during extraction.
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Primary parser ────────────────────────────────────────────────────────────
import pymupdf4llm  # pip install pymupdf4llm

# ── Fallback parser (Docling) ─────────────────────────────────────────────────
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

_GROQ_KEYS = [
    k for k in [
        os.environ.get("GROQ_API_KEY", ""),
        os.environ.get("GROQ_API_KEY_2", ""),
        os.environ.get("GROQ_API_KEY_3", ""),
        os.environ.get("GROQ_API_KEY_4", ""),
    ] if k
]
if not _GROQ_KEYS:
    raise ValueError("No GROQ_API_KEY keys configured")
_GROQ_CLIENTS = [Groq(api_key=k) for k in _GROQ_KEYS]
# ── Failed chunk recovery ────────────────────────────────────────────────────
# Saves chunks that fail extraction so they can be re-processed later with
# improved prompts, different models, or manual review.
# Format: one JSON object per line (JSONL) for easy incremental processing.

import datetime as _dt

_FAILED_CHUNKS_DIR = Path(__file__).parent.parent / "failed_chunks"
_FAILED_CHUNKS_DIR.mkdir(exist_ok=True)

def _save_failed_chunk(chunk_text: str, raw_response: str, extraction_type: str, error: str) -> None:
    """
    Append a failed chunk to a JSONL file for later re-processing.
    
    Saved alongside the extracted DB so it can be re-ingested when the
    extraction pipeline improves (better repair_json, different model, etc.)
    """
    record = {
        "timestamp":       _dt.datetime.now().isoformat(),
        "extraction_type": extraction_type,
        "error":           error[:300],
        "chunk":           chunk_text,
        "raw_response":    raw_response[:2000] if raw_response else "",
    }
    filepath = _FAILED_CHUNKS_DIR / "failed_extractions.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.debug("   📝 Failed chunk saved to %s", filepath)

import random as _rnd

def _groq_json(prompt: str, max_tokens: int = 3072) -> str:
    """Text extraction via Groq with key rotation — 4× TPD limits."""
    clients = list(_GROQ_CLIENTS)
    _rnd.shuffle(clients)
    for client in clients:
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            finish = resp.choices[0].finish_reason
            content = resp.choices[0].message.content.strip()
            if finish == "length":
                logger.warning("   ⚠️  Response truncated (finish_reason=length), length=%d chars", len(content))
            return content
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ("429", "rate", "limit", "tpd", "tpm", "rpm")):
                time.sleep(4)
                continue
            raise
    raise Exception("All Groq keys exhausted")

logger = logging.getLogger(__name__)

# ── Env + Gemini setup ────────────────────────────────────────────────────────
_GEMINI_KEYS = [
    k for k in [
        os.environ.get("GOOGLE_AI_API_KEY", ""),
        os.environ.get("GOOGLE_AI_API_KEY_2", ""),
        os.environ.get("GOOGLE_AI_API_KEY_3", ""),
        os.environ.get("GOOGLE_AI_API_KEY_4", ""),
    ] if k
]
if not _GEMINI_KEYS:
    raise ValueError("No GOOGLE_AI_API_KEY keys configured")

import random as _random
import httpx

_FLASH_MODEL = "models/gemini-3.1-flash-lite"

def _make_genai_client(api_key: str):
    """Create a genai.Client with proper write timeout for image uploads."""
    http_client = httpx.Client(
        timeout=httpx.Timeout(
            connect=30.0,
            write=120.0,   # Image upload — the key fix
            read=120.0,    # Response wait
            pool=10.0,
        ),
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    )
    return genai.Client(
        api_key=api_key,
        http_options={'httpx_client': http_client},
    )

_CLIENTS = [_make_genai_client(k) for k in _GEMINI_KEYS]

def _get_client():
    """Return a randomly selected pre-created client."""
    return _random.choice(_CLIENTS)  # noqa

# ── Shared Gemini rate limiter ────────────────────────────────────────────────
# Fix 5: Replace per-function time.sleep(4) with a thread-safe shared limiter.
#
# Problem: caption_images_with_gemini() runs 3 workers in parallel.  Each
# worker used to call time.sleep(4) independently, so all three could fire
# within the same second → rate-limit burst (429 errors).
#
# Solution: one module-level _last_gemini_call timestamp + threading.Lock().
# _gemini_throttle() is called by EVERY function that touches the Gemini API
# (extract_structured_data, extract_causal_knowledge, _caption_single_image).
# The lock serialises concurrent workers so consecutive calls are always ≥4 s
# apart across ALL threads — not just within a single thread.
#
# At 3 image workers this yields ≤ 15 RPM (well inside the free-tier 15 RPM
# limit) while allowing text-extraction calls to interleave safely.

import threading as _threading

_gemini_lock: _threading.Lock = _threading.Lock()
_last_gemini_call: float = 0.0
_GEMINI_MIN_INTERVAL: float = 4.0          # seconds between Gemini calls


def _gemini_throttle() -> None:
    """
    Block until at least _GEMINI_MIN_INTERVAL seconds have elapsed since the
    last Gemini API call, then update the timestamp.

    Thread-safe: the Lock ensures only one caller proceeds at a time, so
    parallel ThreadPoolExecutor workers never send concurrent bursts.
    """
    global _last_gemini_call
    with _gemini_lock:
        now   = time.time()
        delta = now - _last_gemini_call
        if delta < _GEMINI_MIN_INTERVAL:
            time.sleep(_GEMINI_MIN_INTERVAL - delta)
        _last_gemini_call = time.time()

# ── BGE-M3 embedding model — singleton ───────────────────────────────────────
# SentenceTransformer("BAAI/bge-m3") is the ONLY embedding model used here.
# CrossEncoder / reranker is intentionally NOT loaded in this file — it is only
# needed at query time and lives in crossencoder_reranker.py.
_EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-loaded singleton for BAAI/bge-m3.
    First call downloads the model (~1.1 GB); all subsequent calls return the
    cached in-process instance — no repeated disk I/O.
    Thread-safe for concurrent encode() calls (releases the GIL internally).
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(
            "⚙️  Loading BGE-M3 embedding model '%s' "
            "(first run: ~1–2 min download, then cached)...",
            _EMBEDDING_MODEL_NAME,
        )
        _embedding_model = SentenceTransformer(
            _EMBEDDING_MODEL_NAME,
            device="cpu",   # swap to "cuda" if a GPU is available
        )
        logger.info("✅ BGE-M3 loaded.")
    return _embedding_model


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """
    Embed a list of strings with BGE-M3.

    Returns a list of float lists (JSON-serialisable).
    Embedding dimension is determined by the model at runtime — never hardcoded.
    normalize_embeddings=True means cosine similarity == dot product downstream.

    batch_size=16 is conservative for CPU; raise to 32-64 on machines with
    ≥16 GB RAM or when using a GPU.
    """
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    # embeddings is a numpy ndarray of shape (len(texts), dim).
    # Iterate row-by-row and call .tolist() on each numpy vector so the result
    # is always a plain list[float] regardless of model version or dtype.
    return [vec.tolist() for vec in embeddings]


def embed_single(text: str) -> list[float]:
    """
    Convenience wrapper: embed one string, return one vector.
    Uses the same batch path so model load happens at most once.
    """
    result = embed_texts([text], batch_size=1)
    return result[0] if result else []


# ── Component alias cache ─────────────────────────────────────────────────────
# Stores {canonical_key: np.ndarray} for O(N) cosine lookups during graph build.
# Using a separate dict (not the graph dict itself) avoids re-embedding on every
# lookup and keeps the alias resolution path at O(N) with a single np.dot per key.
# N = number of unique canonical component names seen so far across all PDFs.
# For 10 pump manuals this is typically 30–80 nodes — well within CPU budget.
_component_embeddings: dict[str, np.ndarray] = {}

_ALIAS_SIMILARITY_THRESHOLD = 0.88   # cosine sim ≥ this → treat as alias


def _resolve_component_alias(name: str) -> str:
    """
    Given a raw component name, return the canonical key it belongs to.

    Algorithm:
      1. Embed the incoming name with BGE-M3.
      2. Compute cosine similarity against every vector in _component_embeddings
         using a single np.dot call per entry (O(N), N = known components).
      3. If best similarity ≥ _ALIAS_SIMILARITY_THRESHOLD → return that key
         (the incoming name is an alias of an existing canonical component).
      4. Otherwise → register this name as a new canonical key and return it.

    Side effect: may add a new entry to _component_embeddings.
    """
    if not name or not name.strip():
        return "unknown"

    key = name.lower().strip()

    # If already a known canonical key, return immediately (O(1) fast path).
    if key in _component_embeddings:
        return key

    emb = np.array(embed_single(name), dtype=np.float32)

    best_key: str | None = None
    best_sim: float = 0.0

    for existing_key, existing_emb in _component_embeddings.items():
        sim = float(np.dot(emb, existing_emb))
        if sim > best_sim:
            best_sim = sim
            best_key = existing_key

    if best_sim >= _ALIAS_SIMILARITY_THRESHOLD and best_key is not None:
        logger.debug(
            "   🔗 Alias: '%s' → '%s' (cosine=%.3f)", name, best_key, best_sim
        )
        return best_key

    # New canonical component — register it.
    _component_embeddings[key] = emb
    return key


# ── Quality thresholds ────────────────────────────────────────────────────────
# Minimum characters per page to consider extraction "good"
MIN_CHARS_PER_PAGE = 100
# If more than this fraction of pages are sparse → scanned PDF suspected
SCANNED_PAGE_RATIO_THRESHOLD = 0.4


# ── Pydantic schemas ──────────────────────────────────────────────────────────
#
# Technician-first schema design:
#   • RepairStep        — one numbered action with branching (if_fail_then)
#   • DiagnosticProcedure — primary extraction unit (fault + steps + context)
#   • FaultEntry        — compact fault-library record (symptom → causes → verify → repair)
#   • DiagnosticBranch  — one yes/no branch node in a diagnostic tree
#   • DiagnosticTree    — root symptom + ordered branch list
#   • RepairProcedure   — standalone named repair procedure (e.g. "Replace bearing")
#   • PageExtraction    — top-level Gemini response envelope
#   • ComponentNode     — component graph node (cross-PDF)
#   • SpecRecord        — one numeric specification row
#   • ImageKnowledgeRecord — image caption + relationships
#   • KnowledgeIndexEntry  — cross-link index (procedures × specs × images)
#
# Embedding policy:
#   JSON files contain only embedding_text (human-readable string).
#   Vectors are generated separately and stored in a vector store, NOT in JSON.
#   This keeps JSON files human-readable and avoids bloating them with float arrays.

class RepairStep(BaseModel):
    step_number:     int
    instruction:     str
    expected_result: str = Field(default="", description="What should happen. Empty string if not specified.")
    if_fail_then:    str = Field(default="", description="What to do if it fails. Empty string if not specified.")


class DiagnosticProcedure(BaseModel):
    machine_family: str = Field(description="e.g., Water Pump, Electric Motor, Submersible Pump")
    system:         str = Field(description="e.g., Electrical, Mechanical, Hydraulic, Sealing")
    component:      str = Field(description="The specific part being diagnosed")
    symptoms:       list[str] = Field(description="Farmer-observable symptoms")
    causes:         list[str] = Field(description="Root causes linked to symptoms")
    required_tools: list[str] = Field(default_factory=list)
    step_sequence:  list[RepairStep]
    safety_warnings: list[str] = Field(default_factory=list)
    part_numbers:    list[str] = Field(default_factory=list)
    # ── Causal / relationship fields ──────────────────────────────────────────
    connected_to:  list[str] = Field(
        default_factory=list,
        description="Parts physically connected to this component.",
    )
    if_wrong_installation: list[str] = Field(
        default_factory=list,
        description="Symptoms that appear when installation is wrong or degrades.",
    )
    # ── Provenance (injected by caller, never by Gemini) ──────────────────────
    knowledge_type: str | None = None
    manual_source:  str | None = None
    source_section: str | None = None
    chunk_id:       str | None = None


class FaultEntry(BaseModel):
    """
    Compact fault-library record.
    One entry per distinct symptom cluster observed in the manual.
    Used to answer: 'Farmer says X → what does that mean and what to check?'
    """
    symptom:             str   = Field(description="Farmer-observable symptom phrase")
    problem_description: str   = Field(default="", description="Plain-language explanation of what is happening mechanically")
    likely_causes:       list[str] = Field(default_factory=list)
    understanding:       str   = Field(default="", description="One-sentence mechanic reasoning for why this happens")
    verify:              list[str] = Field(default_factory=list, description="Ordered checks to run first (easiest/cheapest first)")
    repair:              list[str] = Field(default_factory=list, description="Repair actions once cause is confirmed")
    machine_family:      str   = Field(default="")
    component:           str   = Field(default="")
    # Provenance
    manual_source:  str | None = None
    source_section: str | None = None


class DiagnosticBranch(BaseModel):
    """One yes/no decision node in a branching diagnostic tree."""
    question: str
    yes_next: str = Field(default="", description="What to do or check if YES", alias="yes")
    no_next:  str = Field(default="", description="What to do or check if NO",  alias="no")

    model_config = {"populate_by_name": True}


class DiagnosticTree(BaseModel):
    """
    Branching diagnostic tree rooted at one symptom.
    Enables: 'Pump humming but no water' → branch by shaft rotation → branch by suction.
    """
    symptom:       str
    machine_family: str = Field(default="")
    component:      str = Field(default="")
    branches:       list[DiagnosticBranch] = Field(default_factory=list)
    # Provenance
    manual_source:  str | None = None
    source_section: str | None = None


class RepairProcedure(BaseModel):
    """
    Named repair procedure (e.g. 'Replace bearing', 'Re-prime pump').
    Standalone — can be referenced by FaultEntry.repair or DiagnosticTree branch.
    """
    procedure:       str
    component:       str   = Field(default="")
    machine_family:  str   = Field(default="")
    steps:           list[str] = Field(default_factory=list, description="Plain-text ordered steps")
    tools:           list[str] = Field(default_factory=list)
    expected_result: str   = Field(default="", description="What correct completion looks like")
    safety_warnings: list[str] = Field(default_factory=list)
    part_numbers:    list[str] = Field(default_factory=list)
    # Provenance
    manual_source:  str | None = None
    source_section: str | None = None


class PageExtraction(BaseModel):
    """Top-level envelope for all Gemini extraction responses."""
    procedures:         list[DiagnosticProcedure] = Field(default_factory=list)
    fault_entries:      list[FaultEntry]           = Field(default_factory=list)
    diagnostic_trees:   list[DiagnosticTree]       = Field(default_factory=list)
    repair_procedures:  list[RepairProcedure]      = Field(default_factory=list)
    tables:             list[dict] = Field(default_factory=list)

# ── Supporting database schemas ───────────────────────────────────────────────

class ComponentNode(BaseModel):
    """
    One node in the component relationship graph.
    Embedding policy: embedding_text stored in JSON; vector goes to vector store.
    """
    component:       str
    aliases:         list[str] = []
    connected_to:    list[str] = []
    function:        str       = ""   # what this component does (from causal extraction)
    diagram_sources: list[str] = []
    manual_sources:  list[str] = []
    related_faults:  list[str] = []
    related_specs:   list[str] = []
    machine_family:  str       = ""
    embedding_text:  str       = ""   # human-readable; vector computed separately


class SpecRecord(BaseModel):
    """One numeric specification row."""
    component:               str
    parameter:               str   = ""   # e.g. "alignment tolerance", "capacitor rating"
    spec_type:               str   = ""   # legacy field kept for compatibility
    value:                   str
    unit:                    str
    acceptable_range:        str   = ""
    page:                    str   = ""   # section / page reference
    source_page:             str   = ""   # legacy alias
    manual_source:           str
    if_out_of_range:         list[str] = []   # symptoms when spec is violated
    failure_if_out_of_range: list[str] = []   # legacy alias
    repair_actions:          list[str] = []
    embedding_text:          str   = ""


class ImageKnowledgeRecord(BaseModel):
    """
    Image caption record — repair-oriented fields.
    Embedding policy: embedding_text stored in JSON; vector computed separately.
    """
    filename:                str
    caption:                 str   = ""
    diagram_type:            str   = ""
    diagram_type_classifier: str   = "other"
    components_visible:      list[str]  = []   # upgraded field name (request spec)
    components_shown:        list[str]  = []   # legacy alias kept for compatibility
    labels_detected:         list[str]  = []
    arrows_point_to:         list[dict] = []   # [{"label": str, "points_to": str}]
    relationships:           list[dict] = []   # [{"from": str, "to": str, "type": str}]
    connected_relationships: list[dict] = []   # legacy alias
    specifications:          list[str]  = []   # spec values visible in the image
    fault_relevance:         list[str]  = []   # specific faults this image shows
    repair_relevance:        list[str]  = []   # repair tasks this image supports
    assembly_order:          list[str]  = []   # ordered assembly steps if visible
    search_keywords:         list[str]  = []
    manual_source:           str        = ""
    page:                    int        = 0
    section_context:         str        = ""
    surrounding_text:        str        = ""
    embedding_text:          str        = ""


class KnowledgeIndexEntry(BaseModel):
    """Cross-link index — one entry per canonical component."""
    component:            str
    procedure_ids:        list[str] = []
    fault_ids:            list[str] = []
    spec_ids:             list[str] = []
    image_ids:            list[str] = []
    connected_components: list[str] = []
    embedding_text:       str       = ""

def _pymupdf_to_markdown(pdf_path: str) -> str:
    """Convert PDF to markdown using PyMuPDF4LLM (fast, native text layer)."""
    return pymupdf4llm.to_markdown(pdf_path)


def _assess_extraction_quality(md_text: str, pdf_path: str) -> tuple[bool, str]:
    """
    Heuristically decide whether PyMuPDF4LLM extraction is good enough.

    Returns (is_good: bool, reason: str).
    Triggers Docling fallback when:
      - text layer is absent / very sparse (likely scanned PDF)
      - extraction is suspiciously short for the page count
      - table markers are detected but content looks garbled
    """
    import fitz  # PyMuPDF (installed with pymupdf4llm)

    doc = fitz.open(pdf_path)
    page_count = len(doc)
    doc.close()

    if page_count == 0:
        return False, "zero pages"

    total_chars = len(md_text.strip())

    # Very sparse overall → scanned or image-only PDF
    avg_chars = total_chars / page_count
    if avg_chars < MIN_CHARS_PER_PAGE:
        return False, f"sparse text ({avg_chars:.0f} chars/page avg — likely scanned)"

    # Check for garbled table content: pipe characters with very short cell values
    # suggest Docling's table reconstructor would do better
    lines = md_text.splitlines()
    table_lines = [l for l in lines if l.startswith("|")]
    if table_lines:
        # If >30 % of table cells are single characters → extraction is garbled
        cells = [c.strip() for l in table_lines for c in l.split("|") if c.strip()]
        single_char_ratio = sum(1 for c in cells if len(c) == 1) / max(len(cells), 1)
        if single_char_ratio > 0.30:
            return False, f"garbled table cells ({single_char_ratio:.0%} single-char)"

    return True, "ok"


# ── Docling — Fallback parser ─────────────────────────────────────────────────

def _build_docling_converter() -> DocumentConverter:
    """
    Build a Docling converter with OCR enabled.
    FIX #6: do_ocr=True enables Tesseract for scanned PDFs (common in Indian pump manuals).
    generate_page_images=True is required for OCR to function.
    If you hit RAM issues on low-memory servers, set do_ocr=False as a fallback,
    but know that scanned pages will return empty text.
    """
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        generate_page_images=True,   # FIX #6: required for OCR
        do_ocr=True,                 # FIX #6: enables OCR for scanned/image-only pages
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

_docling_converter: DocumentConverter | None = None

def _get_docling_converter() -> DocumentConverter:
    global _docling_converter
    if _docling_converter is None:
        logger.info("⚙️  Loading Docling (fallback, Low-RAM Mode)...")
        _docling_converter = _build_docling_converter()
    return _docling_converter


def _docling_to_markdown(pdf_path: str) -> str:
    """Convert PDF to markdown using Docling (table-aware, slower)."""
    converter = _get_docling_converter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


# ── Hybrid dispatcher ─────────────────────────────────────────────────────────

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Attempt PyMuPDF4LLM first. Fall back to Docling if quality is poor.
    """
    pdf_name = Path(pdf_path).name

    # ── Primary: PyMuPDF4LLM ─────────────────────────────────────────────────
    logger.info("📄 [PRIMARY] PyMuPDF4LLM → %s", pdf_name)
    try:
        md_text = _pymupdf_to_markdown(pdf_path)
    except Exception as e:
        logger.warning("⚠️  PyMuPDF4LLM failed (%s) — switching to Docling", e)
        md_text = ""

    # ── Quality check ─────────────────────────────────────────────────────────
    if md_text:
        is_good, reason = _assess_extraction_quality(md_text, pdf_path)
        if is_good:
            logger.info("   ✅ PyMuPDF4LLM quality OK — proceeding")
            return md_text
        logger.warning("   ⚠️  Quality check failed: %s — falling back to Docling", reason)
    else:
        logger.warning("   ⚠️  Empty output from PyMuPDF4LLM — falling back to Docling")

    # ── Fallback: Docling ─────────────────────────────────────────────────────
    logger.info("📄 [FALLBACK] Docling → %s", pdf_name)
    try:
        return _docling_to_markdown(pdf_path)
    except Exception as e:
        logger.error("❌ Docling also failed: %s", e)
        return ""


# ── Section filter ───────────────────────────────────────────────────────────
#
# Two-tier decision:
#   TIER 1 — Header match  (cheap, runs on section title only)
#     • HARD-SKIP headers  → discard immediately, no content scan needed
#     • HARD-KEEP headers  → accept immediately
#   TIER 2 — Content scan  (only for ambiguous headers)
#     • Scan first 600 chars for fault-signal density
#     • Require ≥ MIN_FAULT_SIGNALS distinct keep-signals to accept
#
# This eliminates false-positives from generic sections that happen to contain
# one stray keyword (e.g. a "Warning" footnote inside a warranty page).

_KEEP_KEYWORDS: list[str] = [
    "troubleshooting", "breakdown",  "failure",    "maintenance",
    "repair",          "fault",      "problem",    "cause",
    "remedy",          "cavitation", "overhaul",   "diagnostic",
    "inspection",      "defect",     "leak",       "noise",
    "vibration",       "overheating","symptom",    "corrective action",
    "does not start",  "will not",   "excessive",  "low pressure",
    "check valve",     "worn",       "seized",     "blocked",
    "motor",           "winding",    "insulation", "capacitor",   "impeller",
    "shaft",           "seal",       "bearing",    "coupling",    "discharge",
    "suction",         "priming",    "voltage",    "current",
    "overload",        "thermostat", "contactor",  "starter",
    "strainer",        "foot valve", "non-return", "delivery valve",
    "operation problem","commissioning","no discharge","insufficient head",
    "reverse rotation","single phasing","air locked","does not prime",
    "install", "assembly", "assemble", "commission", "coupling",
    "alignment", "torque", "clearance", "tolerance",
    "terminal", "capacitor", "rating", "wiring", "connection",
    "priming", "lubrication", "oil level", "adjustment", "calibration",
    "tighten", "sight glass", "fill level", "align", "keyway",  
]

_SKIP_KEYWORDS: list[str] = [
    "foreword",            "history",           "warranty",
    "table of contents",   "contents",          "standards",
    "conversion table",    "unit conversion",   "formula",
    "specification only",  "about this manual",
    "how to use",          "abbreviation",      "glossary",
    "index",               "spare parts list",  "parts catalogue",
    "ordering parts",
]
_HARD_SKIP_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(foreword|warranty|contents|glossary|index|abbreviations?)\b", re.I),
    re.compile(r"\bparts\s+(list|catalogue|catalog)\b", re.I),
    re.compile(r"\bunit\s+conversion\b", re.I),
    re.compile(r"^introduction\s*$", re.I),
]
_HARD_KEEP_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(troubleshoot(ing)?|fault|diagnostic|overhaul)\b", re.I),
    re.compile(r"\b(repair|maintenance|inspection|breakdown)\b", re.I),
    re.compile(r"\b(does not|will not|fail(ure|s)?|problem)\b", re.I),
    re.compile(r"\b(operation\s+problem|commissioning|corrective|remedy)\b", re.I),
    re.compile(r"\b(not\s+start|no\s+flow|no\s+discharge|overload|cavit)\b", re.I),
    re.compile(r"\b(install(ation)?|assembl[ey](ing)?|commission(ing)?)\b", re.I),
    re.compile(r"\b(coupling|alignment|torque|clearance|tolerance)\b", re.I),
    re.compile(r"\b(wiring|connection|terminal|capacitor|rating|sizing)\b", re.I),
    re.compile(r"\b(priming|filling|lubrication|oil\s+level|coolant\s+level)\b", re.I),
    re.compile(r"\b(adjustment|calibration|tensioning|bleeding|setting)\b", re.I),
]
_MIN_FAULT_SIGNALS = 1

def _is_relevant_section(section: str, content: str) -> bool:
    """
    Return True if the chunk should be sent to Gemini.

    Tier 1 — header-only patterns (O(1), regex):
      • Any HARD_SKIP pattern in header and no HARD_KEEP → discard
      • Any HARD_KEEP pattern in header                  → accept

    Tier 2 — content signal density (for ambiguous headers):
      • Count distinct keep-keywords in first 600 chars of content
      • Accept only if count ≥ _MIN_FAULT_SIGNALS
      • If a skip keyword also fires and signal count is 0 → discard
    """
    # Tier 1
    hard_skip = any(p.search(section) for p in _HARD_SKIP_PATTERNS)
    hard_keep = any(p.search(section) for p in _HARD_KEEP_PATTERNS)

    if hard_skip and not hard_keep:
        return False
    if hard_keep:
        return True

    # Tier 2 — ambiguous header, scan content
    # FIX #1: Widened from 600 → 1200 chars; diagnostic keywords often appear later in paragraph
    haystack = (section + " " + content[:1200]).lower()
    signals  = sum(1 for kw in _KEEP_KEYWORDS if kw in haystack)
    has_skip = any(kw in haystack for kw in _SKIP_KEYWORDS)

    if has_skip and signals == 0:
        return False           # boilerplate with zero fault signals
    return signals >= _MIN_FAULT_SIGNALS

_INSTALL_SIGNALS = re.compile(
    r"\b(install(ation)?|assembl[ey](ing)?|commission(ing)?|coupling|"
    r"alignment|torque|clearance|tolerance|wiring\s+diagram|"
    r"capacitor\s+rating|oil\s+grade|fill\s+level|priming\s+procedure|"
    r"terminal\s+connection|commissioning|specification)\b",
    re.I,
)

_FAULT_SIGNALS = re.compile(
    r"\b(troubleshoot|fault|symptom|cause|remedy|repair|does\s+not|"
    r"will\s+not|failure|breakdown|defect|diagnos)\b",
    re.I,
)

def _is_installation_section(section: str, content: str) -> bool:
    haystack = section + " " + content[:800]
    has_install = bool(_INSTALL_SIGNALS.search(haystack))
    has_fault   = bool(_FAULT_SIGNALS.search(haystack))
    return has_install and not has_fault

# ── Chunking ──────────────────────────────────────────────────────────────────

# Secondary splitter: caps any chunk that survived header-splitting at 5 000 chars
# (~1 250 tokens), with 500-char overlap to preserve cross-boundary context.
# Smaller chunks keep each repair procedure isolated — large chunks mix unrelated
# faults, dilute symptom context, and increase hallucination risk.
_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5_000,
    chunk_overlap=500,
)

def chunk_markdown(md_text: str) -> list[dict]:
    """
    Three-stage chunking pipeline.

    Stage 1 — MarkdownHeaderTextSplitter:
        Splits on H1/H2/H3 boundaries and captures the section title.
    Stage 2 — Intelligent section filter (_is_relevant_section):
        Drops boilerplate sections (foreword, warranty, conversion tables, …)
        and keeps only diagnostic/repair content. Reduces Gemini calls and
        improves extraction quality in one step.
    Stage 3 — RecursiveCharacterTextSplitter:
        Cuts any oversized chunk (wiring tables, appendices, etc.) down to
        ≤5 000 chars so Gemini receives focused, single-procedure context.

    Returns a list of dicts:
        { "text": str, "section": str, "chunk_id": str }
    """
    import hashlib as _hl

    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_docs = header_splitter.split_text(md_text)

    kept = skipped = 0
    final_chunks: list[dict] = []

    for doc in header_docs:
        content = doc.page_content.strip()
        if len(content) < 120:          # too short to contain a real procedure
            continue

        # Pull the deepest header available as the section label
        meta    = doc.metadata
        section = meta.get("H3") or meta.get("H2") or meta.get("H1") or "unknown"

        # Stage 2: filter
        if not _is_relevant_section(section, content):
            logger.debug("   ⏭  Skipping irrelevant section: '%s'", section)
            skipped += 1
            continue
        kept += 1

        # Stage 3: secondary split for large chunks
        sub_texts = _recursive_splitter.split_text(content)
        for sub in sub_texts:
            if len(sub.strip()) < 120:
                continue
            chunk_id = _hl.md5(sub.encode()).hexdigest()[:12]
            final_chunks.append({
                "text":     sub,
                "section":  section,
                "chunk_id": chunk_id,
            })

    logger.info(
        "   🔍 Section filter: %d kept, %d skipped → %d final chunks",
        kept, skipped, len(final_chunks),
    )
    return final_chunks

def extract_structured_data(chunk: str, retries: int = 3) -> dict:
    """
    Extract repair intelligence from troubleshooting / fault / maintenance text.

    Returns a PageExtraction-compatible dict with four top-level arrays:
      • procedures       — detailed step-by-step diagnostic procedures
      • fault_entries    — compact fault library (symptom → causes → verify → repair)
      • diagnostic_trees — branching yes/no diagnostic logic trees
      • repair_procedures — named standalone repair procedures

    Technician-first design: the prompt forces Gemini to reason like an experienced
    field mechanic, not like a document summariser.
    """
    prompt = f"""You are a senior field repair technician (20 yrs) for agricultural machinery and equipment. Extract REPAIR INTELLIGENCE like a mechanic, not a document summary: what does the operator/farmer see, why does it happen, what do you check first.

Return ONE raw JSON object. No markdown, no preamble, no trailing text.
Keys: procedures, fault_entries, diagnostic_trees, repair_procedures — all arrays, empty array if no content.

FIELD RULE — machine_family and system are SINGLE-SELECT, inferred from context, NOT a closed list:
- machine_family: read the text and name the machine type it describes. Examples (not exhaustive): Tractor, Harvester, Thresher, Generator, Water Pump, Submersible Pump, Electric Motor, Diesel Engine, Hydraulic Unit, Rotavator, Power Tiller. If the text is about something else, name THAT instead.
- system: read the text and name the subsystem involved. Examples (not exhaustive): Electrical, Mechanical, Hydraulic, Fuel, Cooling, Transmission, Lubrication, Sealing, Pump, Steering, Brake, Clutch, Engine, Starting. If the text describes a different subsystem, name THAT instead.
- BOTH fields take exactly ONE value, however many examples are listed above.
WRONG: "Tractor | Diesel Engine"   RIGHT: "Diesel Engine"  (pick the single best match — never join with "|", "/", "or", or commas)

## procedures — one per distinct fault/component, exhaustive steps, all numeric specs
{{"machine_family":"<inferred from text, single value>","system":"<inferred from text, single value>","component":"<specific part>","symptoms":["<operator-observable>"],"causes":["<root cause>"],"required_tools":["<tool+spec>"],"step_sequence":[{{"step_number":1,"instruction":"<action incl. specs/voltages/tolerances>","expected_result":"<correct reading>","if_fail_then":"<next branch, empty if none>"}}],"safety_warnings":["<every adjacent Warning/Caution/NOTE>"],"part_numbers":["<P/N if given>"],"connected_to":["<physically connected parts>"]}}

## fault_entries — one per distinct operator-observable symptom (compact fault library)
{{"symptom":"<exact phrase>","problem_description":"<1 sentence>","likely_causes":["<most likely first>"],"understanding":"<1 sentence reasoning>","verify":["<cheapest/no-tool check first>"],"repair":["<action once cause confirmed>"],"machine_family":"<same rule as above>","component":"<part>"}}

## diagnostic_trees — ONLY when text has IF/THEN branching logic
{{"symptom":"<root symptom>","machine_family":"<same rule as above>","component":"<part>","branches":[{{"question":"<yes/no question>","yes":"<next if yes>","no":"<next if no>"}}]}}

## repair_procedures — named standalone repair actions
{{"procedure":"<name e.g. Replace Mechanical Seal>","component":"<part>","machine_family":"<same rule as above>","steps":["<step>"],"tools":["<tool>"],"expected_result":"<correct outcome>","safety_warnings":["<warning>"],"part_numbers":["<P/N>"]}}

RULES (all keys):
1. Never invent content not in the text.
2. Symptoms = operator-observable. Bad: "impedance mismatch". Good: "motor hums but does not rotate".
3. One object per distinct fault/component — never merge.
4. Copy numeric specs verbatim: 25µF, 0.3–0.5mm, 230V ±10%.
5. Fault table rows (Symptom|Cause|Remedy) → one fault_entry per row.
6. IF/ELSE logic present → also populate diagnostic_trees.
7. step_sequence with >1 if_fail_then → also add a diagnostic_tree.
8. likely_causes/verify: most-probable/cheapest check first.
9. No fault/repair content → return {{"procedures":[],"fault_entries":[],"diagnostic_trees":[],"repair_procedures":[]}}.
10. SAFETY RULE: Ignore purely precautionary advice (wear gloves, eye protection, loose clothing). Still extract safety instructions that directly explain machine failure, diagnosis, or repair (e.g. "never run pump dry", "do not start with valve closed"). Merge repetitive safety instructions about the same component into a single entry.
--- MANUAL SECTION ---
{chunk}"""
    for attempt in range(retries):
        try:
            time.sleep(5.0)
            text = _groq_json(prompt, max_tokens=3072)
            logger.debug(
                "   📏 Response: %d chars | First 150: %s... | Last 400: ...%s",
                len(text), text[:150], text[-400:]
            )
            if not text:
                return {"procedures": [], "fault_entries": [], "diagnostic_trees": [], "repair_procedures": []}
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"):     text = text[3:]
            if text.endswith("```"):       text = text[:-3]
            text = text.strip()
            parsed    = repair_json(text)
            validated = PageExtraction(**parsed)
            logger.debug(
                " 📊 Objects: %d procedures, %d fault_entries, %d trees, %d repair_procs",
                len(validated.procedures),
                len(validated.fault_entries),
                len(validated.diagnostic_trees),
                len(validated.repair_procedures),
            )
            return validated.model_dump()
        except json.JSONDecodeError as e:
            # JSON repair failed — log the raw text for debugging, then retry
            logger.warning(
                "⚠️ JSON parse error on chunk (attempt %d/%d): %s\n"
                "   Raw text (first 300 chars): %s",
                attempt + 1, retries, str(e)[:100], text[:300] if 'text' in locals() else "N/A"
            )
            _save_failed_chunk(chunk, text if 'text' in locals() else "", "structured", str(e))
            if attempt < retries - 1:
                wait_time = 4 * (attempt + 1)
                logger.info("   🔄 Retrying JSON extraction in %ds...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("❌ JSON extraction failed after %d retries", retries)
                return {"procedures": [], "fault_entries": [], "diagnostic_trees": [], "repair_procedures": []}

        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ("429", "quota", "503", "unavailable", "time out", "timed out", "connection")):
                wait_time = 8 * (attempt + 1)
                logger.warning("⏳ Rate limit hit. Waiting %ds before retry %d/%d...",
                               wait_time, attempt + 1, retries)
                time.sleep(wait_time)
            else:
                logger.error("❌ Unexpected extraction error: %s", e)
                _save_failed_chunk(chunk, text if 'text' in locals() else "", "structured", str(e))
                return {"procedures": [], "fault_entries": [], "diagnostic_trees": [], "repair_procedures": []}
    logger.error("❌ Failed to extract chunk after %d retries.", retries)
    return {"procedures": [], "fault_entries": [], "diagnostic_trees": [], "repair_procedures": []}

def extract_causal_knowledge(chunk: str, retries: int = 3) -> dict:
    """
    Extract CAUSAL REPAIR KNOWLEDGE from installation / assembly / specification text.

    Strategy — causal inversion:
      Installation steps describe the CORRECT state.
      Repair knowledge is the INVERSE: what breaks when that state degrades.
      "Align shaft ±0.05 mm" → if not: vibration, bearing noise, seal leak.
      "Capacitor rated 25µF ±5%" → if degraded: motor hums but won't rotate.
      "Foot valve 600 mm below water table" → if not: air lock, no prime.

    Returns the same four-key dict as extract_structured_data() so the same
    normalization and save pipeline applies downstream.
    """
    prompt = f"""You are a senior agricultural equipment service engineer (20 yrs). Installation knowledge is the foundation of repair knowledge: every spec, tolerance, and assembly step encodes what breaks when it's wrong. For each spec/tolerance/wiring/assembly step below, ask: "if this is wrong now or degrades over time, what does the operator observe?"

Return ONE raw JSON object. No markdown, no preamble.
Keys: procedures, fault_entries, diagnostic_trees, repair_procedures, tables — all arrays, empty if no content.

FIELD RULE — machine_family and system are SINGLE-SELECT, inferred from context, NOT a closed list:
- machine_family: read the text and name the machine type it describes. Examples (not exhaustive): Tractor, Harvester, Thresher, Generator, Water Pump, Submersible Pump, Electric Motor, Diesel Engine, Hydraulic Unit, Rotavator, Power Tiller. If the text is about something else, name THAT instead.
- system: read the text and name the subsystem involved. Examples (not exhaustive): Electrical, Mechanical, Hydraulic, Fuel, Cooling, Transmission, Lubrication, Sealing, Pump, Steering, Brake, Clutch, Engine, Starting. If the text describes a different subsystem, name THAT instead.
- BOTH fields take exactly ONE value, however many examples are listed above.
WRONG: "Tractor | Diesel Engine"   RIGHT: "Diesel Engine"  (never join with "|", "/", "or", or commas)

## procedures — one per distinct spec/component; include how to measure AND correct it
{{"machine_family":"<inferred, single value>","system":"<inferred, single value>","component":"<part>","symptoms":["<operator-observable when spec is wrong>"],"causes":["<root cause>"],"required_tools":["<measurement tool>"],"step_sequence":[{{"step_number":1,"instruction":"<check/repair action incl. spec value>","expected_result":"<correct state incl. spec>","if_fail_then":"<action if exceeded>"}}],"safety_warnings":[],"part_numbers":[],"connected_to":["<physically connected parts>"],"if_wrong_installation":["<symptom caused by this exact wrong state, only>"]}}

## fault_entries — one per distinct operator-observable symptom
{{"symptom":"<exact phrase>","problem_description":"<1 sentence>","likely_causes":["<most likely first>"],"understanding":"<1 sentence reasoning>","verify":["<cheapest/no-tool check first>"],"repair":["<action once cause confirmed>"],"machine_family":"<same rule as above>","component":"<part>"}}

## diagnostic_trees — ONLY when text has IF/THEN branching logic
{{"symptom":"<root symptom>","machine_family":"<same rule as above>","component":"<part>","branches":[{{"question":"<yes/no question>","yes":"<next if yes>","no":"<next if no>"}}]}}

## repair_procedures — named standalone repair actions
{{"procedure":"<name>","component":"<part>","machine_family":"<same rule as above>","steps":["<step>"],"tools":["<tool>"],"expected_result":"<correct outcome>","safety_warnings":["<warning>"],"part_numbers":["<P/N>"]}}

## tables — every table in the text, one record per table
table_type must be ONE of: electrical_specs, bearing_specs, torque_table, lubrication_schedule, parts_list, wiring_table, performance_curve, maintenance_schedule, lifting_capacity, cable_specs, weight_table, other
{{"table_type":"<one value above>","page_number":0,"manual_source":"","headers":["<col>"],"rows":[{{"<header>":"<value>"}}],"raw_text_excerpt":"<verbatim table text>"}}

WHAT TO EXTRACT (>=1 procedure + >=1 fault_entry each, if present):
- tolerances/clearances → vibration, noise, overheating, seal leak
- electrical specs (µF, V, wiring) → motor hums, reverse rotation, trips breaker
- fluid specs (oil/coolant/grease) → seizure, overheating, power loss
- installation sequence (priming/bleeding/commissioning) → air lock, cavitation, no prime
- component connections (shaft to coupling to motor) → vibration, noise, seal leak

RULES:
1. Never invent a spec not in the text — copy numeric values verbatim (e.g. 25µF).
2. Symptoms = operator-observable, never jargon.
3. One procedure per distinct component/spec — never merge.
4. The exact numeric spec must appear in both instruction and expected_result.
5. if_wrong_installation lists only symptoms actually caused by this specific wrong state.
6. SAFETY RULE: Ignore purely precautionary advice (wear gloves, eye protection, loose clothing). Still extract safety instructions that directly explain machine failure, diagnosis, or repair (e.g. "never run pump dry", "do not start with valve closed"). Merge repetitive safety instructions about the same component into a single entry.
--- TEXT ---
{chunk}"""
    for attempt in range(retries):
        try:
            time.sleep(5.0)
            text = _groq_json(prompt, max_tokens=3072)
            logger.debug(
                "   📏 Response: %d chars | First 150: %s... | Last 400: ...%s",
                len(text), text[:150], text[-400:]
            )
            if not text:
                return {"procedures": [], "fault_entries": [], "diagnostic_trees": [], "repair_procedures": []}
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"):     text = text[3:]
            if text.endswith("```"):       text = text[:-3]
            text = text.strip()
            parsed    = repair_json(text)
            validated = PageExtraction(**parsed)
            logger.debug(
                "   📊 Objects: %d procedures, %d fault_entries, %d trees, %d repair_procs",
                len(validated.procedures),
                len(validated.fault_entries),
                len(validated.diagnostic_trees),
                len(validated.repair_procedures),
            )
            return validated.model_dump()
        except json.JSONDecodeError as e:
            # JSON repair failed — log the raw text for debugging, then retry
            logger.warning(
                "⚠️ JSON parse error on causal chunk (attempt %d/%d): %s\n"
                "   Raw text (first 300 chars): %s",
                attempt + 1, retries, str(e)[:100], text[:300] if 'text' in locals() else "N/A"
            )
            _save_failed_chunk(chunk, text if 'text' in locals() else "", "causal", str(e))
            if attempt < retries - 1:
                wait_time = 4 * (attempt + 1)
                logger.info("   🔄 Retrying causal JSON extraction in %ds...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("❌ Causal JSON extraction failed after %d retries", retries)
                return {"procedures": [], "fault_entries": [], "diagnostic_trees": [], "repair_procedures": []}

        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ("429", "quota", "503", "unavailable", "time out", "timed out", "connection")):
                wait_time = 8 * (attempt + 1)
                logger.warning("⏳ Rate limit. Waiting %ds (attempt %d/%d)...",
                               wait_time, attempt + 1, retries)
                time.sleep(wait_time)
            else:
                logger.error("❌ Causal extraction failed: %s", e)
                _save_failed_chunk(chunk, text if 'text' in locals() else "", "causal", str(e))
                return {"procedures": [], "fault_entries": [], "diagnostic_trees": [], "repair_procedures": []}
    logger.error("❌ Causal extraction: failed after %d retries.", retries)
    return {"procedures": [], "fault_entries": [], "diagnostic_trees": [], "repair_procedures": []}

# ── Normalization layer ──────────────────────────────────────────────────────
#
# Applied to every procedure dict BEFORE it enters the master database.
# Goals:
#   • Consistent casing on categorical fields (machine_family, system, component)
#   • Strip whitespace / control characters from all string values
#   • Normalize Unicode (NFKC) so accented chars from OCR match clean text
#   • Canonicalize common synonyms so RAG retrieval doesn't fragment results
#   • Drop empty / placeholder strings from list fields
#   • Ensure step_number is sequential (Gemini sometimes resets to 1)

_SYSTEM_SYNONYMS: dict[str, str] = {
    "fuel system":        "Fuel",
    "fueling":            "Fuel",
    "electrical system":  "Electrical",
    "electrics":          "Electrical",
    "hydraulic system":   "Hydraulic",
    "hydraulics":         "Hydraulic",
    "cooling system":     "Cooling",
    "coolant":            "Cooling",
    "transmission system":"Transmission",
    "gearbox":            "Transmission",
    "lubrication system": "Lubrication",
    "lube":               "Lubrication",
    "engine":             "Engine",
    "starting system":    "Starting",
    "starter":            "Starting",
    # FIX #4: Added pump/motor-specific system synonyms
    "pump":               "Pump",
    "pumping system":     "Pump",
    "pump operation":     "Pump",
    "sealing":            "Sealing",
    "sealing system":     "Sealing",
    "motor":              "Motor",
    "installation":       "Installation",
}

# ── FIX #4: Machine family normalization ──────────────────────────────────────
# Gemini invents different names for the same machine family. Without this,
# RAG filters like {"machine_type": "water_pump"} miss entries stored as "Pump".

_MACHINE_FAMILY_SYNONYMS: dict[str, str] = {
    "pump":                              "Water Pump",
    "centrifugal pump":                  "Water Pump",
    "mono-block pump":                   "Water Pump",
    "monoblock pump":                    "Water Pump",
    "centrifugal mono-block pump set":   "Water Pump",
    "centrifugal monoblock pump":        "Water Pump",
    "water pump":                        "Water Pump",
    "submersible pump":                  "Submersible Pump",
    "submersible":                       "Submersible Pump",
    "electric motor":                    "Electric Motor",
    "motor":                             "Electric Motor",
    "engine":                            "Diesel Engine",
    "diesel engine":                     "Diesel Engine",
    "machine":                           "Agricultural Equipment",
    "agricultural equipment":            "Agricultural Equipment",
    "hydraulic unit":                    "Hydraulic Unit",
    "tractor":                           "Tractor",
    "harvester":                         "Harvester",
    "thresher":                          "Thresher",
    "generator":                         "Generator",
}

# Maps canonical machine_family → snake_case machine_type for RAG filter compatibility
_FAMILY_TO_MACHINE_TYPE: dict[str, str] = {
    "Water Pump":           "water_pump",
    "Submersible Pump":     "submersible_pump",
    "Electric Motor":       "electric_motor",
    "Diesel Engine":        "diesel_engine",
    "Tractor":              "tractor",
    "Harvester":            "harvester",
    "Thresher":             "thresher",
    "Generator":            "generator",
    "Hydraulic Unit":       "hydraulic_unit",
    "Agricultural Equipment": "universal",
}

# ── Symptom canonicalization ──────────────────────────────────────────────────
#
# Maps surface-level symptom variants to a stable canonical key.
# Without this, "motor won't start" / "fails to start" / "does not start"
# create three separate DB entries and fragment vector retrieval.
#
# Rules:
#   • Keys must be lowercase (matching is done on lowercased input)
#   • Values are snake_case canonical IDs stored alongside the raw symptom
#   • Add new mappings freely — the pipeline never removes raw text, it only
#     adds a `symptom_canonical` field for deduplication / search

SYMPTOM_SYNONYMS: dict[str, str] = {
    # ── Start failures ────────────────────────────────────────────────────────
    "motor won't start":          "motor_no_start",
    "motor will not start":       "motor_no_start",
    "motor does not start":       "motor_no_start",
    "fails to start":             "motor_no_start",
    "engine won't start":         "motor_no_start",
    "engine will not start":      "motor_no_start",
    "engine does not start":      "motor_no_start",
    "unit will not start":        "motor_no_start",
    "pump will not start":        "motor_no_start",
    "pump does not start":        "motor_no_start",
    # ── Low / no output ──────────────────────────────────────────────────────
    "low pressure":               "low_output_pressure",
    "insufficient pressure":      "low_output_pressure",
    "pressure too low":           "low_output_pressure",
    "low flow":                   "low_output_flow",
    "insufficient flow":          "low_output_flow",
    "no flow":                    "no_output_flow",
    "pump not delivering":        "no_output_flow",
    # ── Overheating ───────────────────────────────────────────────────────────
    "overheating":                "overheating",
    "runs hot":                   "overheating",
    "excessive heat":             "overheating",
    "temperature too high":       "overheating",
    # ── Noise / vibration ─────────────────────────────────────────────────────
    "excessive noise":            "abnormal_noise",
    "unusual noise":              "abnormal_noise",
    "knocking noise":             "abnormal_noise",
    "rattling":                   "abnormal_noise",
    "vibration":                  "excessive_vibration",
    "excessive vibration":        "excessive_vibration",
    # ── Leaks ─────────────────────────────────────────────────────────────────
    "oil leak":                   "oil_leak",
    "leaking oil":                "oil_leak",
    "fuel leak":                  "fuel_leak",
    "leaking fuel":               "fuel_leak",
    "coolant leak":               "coolant_leak",
    "water leak":                 "coolant_leak",
    # ── Power loss ────────────────────────────────────────────────────────────
    "loss of power":              "power_loss",
    "power loss":                 "power_loss",
    "reduced power":              "power_loss",
    "engine lacks power":         "power_loss",
}


def canonicalize_symptom(raw_symptom: str) -> str | None:
    """
    Return the canonical symptom key for a raw symptom string, or None if
    no mapping exists. Matching is case-insensitive and strips punctuation.

    Example:
        canonicalize_symptom("Motor Won't Start")  →  "motor_no_start"
        canonicalize_symptom("bearing failure")    →  None
    """
    normalized = raw_symptom.lower().strip().rstrip(".")
    return SYMPTOM_SYNONYMS.get(normalized)

_PLACEHOLDER_RE = re.compile(
    r"^(\.\.\.|n/?a|none|not specified|empty|unknown|null|-)$", re.I
)


def _clean_str(value: str) -> str:
    """Strip, NFKC-normalize, collapse internal whitespace."""
    value = unicodedata.normalize("NFKC", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _clean_list(items: list) -> list[str]:
    """Clean each string item; drop blanks and placeholder values."""
    out = []
    for item in items:
        s = _clean_str(str(item))
        if s and not _PLACEHOLDER_RE.match(s):
            out.append(s)
    return out


# ── Confidence scoring ────────────────────────────────────────────────────────
#
# Assigns a 0.0–1.0 score to each extracted procedure based on how complete
# and trustworthy it looks. Used later for ranking search results and filtering
# low-quality entries before they enter the vector store.
#
# Scoring components (each weighted, sum ≤ 1.0):
#   symptom_score   : 0.25  — at least one specific symptom present
#   cause_score     : 0.20  — at least one cause present
#   step_score      : 0.25  — step count (saturates at 5+ steps → full score)
#   source_score    : 0.15  — source quality proxy (section title signals)
#   field_score     : 0.15  — completeness of optional fields (tools, warnings)

_HIGH_QUALITY_SECTION_SIGNALS = re.compile(
    r"\b(troubleshoot|fault|diagnostic|overhaul|repair|maintenance)\b", re.I
)


def _compute_confidence(proc: dict) -> float:
    """
    Return a confidence score in [0.0, 1.0] for a normalized procedure dict.
    Higher = more trustworthy / complete extraction.
    """
    score = 0.0

    # ── Symptom presence (0.25) ───────────────────────────────────────────────
    symptoms = proc.get("symptoms") or []
    if symptoms:
        # Bonus if at least one symptom is specific (>3 words)
        specific = any(len(s.split()) > 3 for s in symptoms)
        score += 0.20 if not specific else 0.25

    # ── Cause presence (0.20) ─────────────────────────────────────────────────
    causes = proc.get("causes") or []
    if causes:
        score += 0.20

    # ── Step count (0.25, saturates at 5 steps) ───────────────────────────────
    steps = proc.get("step_sequence") or []
    n_steps = len(steps)
    if n_steps > 0:
        # 1 step → 0.05, 2 → 0.10, …, 5+ → 0.25
        score += min(n_steps / 5, 1.0) * 0.25

    # ── Source quality proxy (0.15) ───────────────────────────────────────────
    section = proc.get("source_section") or ""
    if _HIGH_QUALITY_SECTION_SIGNALS.search(section):
        score += 0.15

    # ── Optional field completeness (0.15) ────────────────────────────────────
    # Up to 0.05 each for tools, warnings, part numbers
    has_tools    = bool(proc.get("required_tools"))
    has_warnings = bool(proc.get("safety_warnings"))
    has_parts    = bool(proc.get("part_numbers"))
    score += sum([has_tools, has_warnings, has_parts]) * 0.05

    return round(min(score, 1.0), 3)


def _normalize_procedure(proc: dict) -> dict:
    """
    Return a normalized copy of a raw procedure dict.
    Does NOT mutate the input.
    """
    p = dict(proc)

    # ── Categorical string fields ─────────────────────────────────────────────
    for field in ("machine_family", "system", "component"):
        raw = _clean_str(str(p.get(field) or ""))
        p[field] = raw.title() if raw else "Unknown"

    # FIX #4: Canonicalize machine_family so RAG filters work correctly.
    # Without this, "Pump", "Centrifugal Pump", "Motor" all stay as separate
    # families and are missed by ChromaDB machine_type filters.
    family_lower = p["machine_family"].lower()
    p["machine_family"] = _MACHINE_FAMILY_SYNONYMS.get(family_lower, p["machine_family"])

    # FIX #4: Add machine_type (snake_case) field that matches RAG filter keys
    p["machine_type"] = _FAMILY_TO_MACHINE_TYPE.get(p["machine_family"], "universal")

    # Canonicalize system name
    sys_lower = p["system"].lower()
    p["system"] = _SYSTEM_SYNONYMS.get(sys_lower, p["system"])

    # ── List fields ───────────────────────────────────────────────────────────
    for field in ("symptoms", "causes", "required_tools", "safety_warnings", "part_numbers"):
        p[field] = _clean_list(p.get(field) or [])

    # Canonicalize symptoms: annotate each symptom with its canonical key so
    # deduplication and vector search operate on stable identifiers, while
    # preserving the original text for human readability.
    p["symptoms_canonical"] = [
        canonicalize_symptom(s) or s.lower().strip()
        for s in p["symptoms"]
    ]

    # ── Step sequence ─────────────────────────────────────────────────────────
    steps = p.get("step_sequence") or []
    cleaned_steps = []
    for i, step in enumerate(steps, start=1):
        cleaned_steps.append({
            "step_number":     i,                                    # re-sequence
            "instruction":     _clean_str(str(step.get("instruction") or "")),
            "expected_result": _clean_str(str(step.get("expected_result") or "")),
            "if_fail_then":    _clean_str(str(step.get("if_fail_then") or "")),
        })
    p["step_sequence"] = cleaned_steps

    # ── Provenance strings ────────────────────────────────────────────────────
    for field in ("manual_source", "source_section", "chunk_id"):
        p[field] = _clean_str(str(p.get(field) or ""))

    # ── New causal fields ─────────────────────────────────────────────────────
    p["connected_to"]          = _clean_list(p.get("connected_to") or [])
    p["if_wrong_installation"] = _clean_list(p.get("if_wrong_installation") or [])

    # ── Confidence score (Issue 3) ────────────────────────────────────────────
    # Computed last so it has access to all normalized fields above.
    p["confidence_score"] = _compute_confidence(p)

    # ── Embedding policy: remove raw vector, keep embedding_text only ─────────
    # Vectors are computed at index time and stored in the vector store.
    # Removing them here keeps the JSON human-readable.
    p.pop("embedding", None)

    return p


# ── Specification extractor ───────────────────────────────────────────────────
# Parses numeric specifications out of normalised procedures into SpecRecord
# dicts for the spec_database.json. Runs entirely locally — zero Gemini calls.
# Called for every procedure (both knowledge types); causal_inferred procs are
# the richest source but direct_repair procs also contain measurements in steps.

_SPEC_PATTERNS: list[tuple[str, str, str]] = [
    # (spec_type, regex_pattern, canonical_unit)
    # Regex captures the numeric value; unit is the canonical label stored in JSON.
    ("capacitor_rating", r"(\d+(?:\.\d+)?)\s*(?:µ[Ff]|uF|microfarad)", "µF"),
    ("voltage",          r"(\d+(?:\.\d+)?)\s*[Vv](?:olt)?s?(?:\s*[A-Z]{2})?", "V"),
    ("current",          r"(\d+(?:\.\d+)?)\s*[Aa](?:mp(?:ere)?s?)?(?:\b)", "A"),
    ("torque",           r"(\d+(?:\.\d+)?)\s*(?:N[\s·\-]?m|Nm|newton[\s-]?met)", "N·m"),
    ("clearance",        r"(\d+(?:\.\d+)?)\s*(?:mm|millim(?:etre|eter))", "mm"),
    ("pressure",         r"(\d+(?:\.\d+)?)\s*(?:bar|psi|kPa|MPa)", "bar"),
    ("temperature",      r"(\d+(?:\.\d+)?)\s*°?\s*[Cc](?:\b)", "°C"),
    ("speed",            r"(\d{3,5})\s*(?:rpm|RPM|r\.p\.m\.?)", "rpm"),
    ("flow_rate",        r"(\d+(?:\.\d+)?)\s*(?:l/s|l/min|lpm|m³/h|lps)", "l/min"),
    ("depth",            r"(\d+(?:\.\d+)?)\s*(?:m(?:etre|eter)?s?)(?:\s+(?:below|depth|submerg))", "m"),
]

_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[–—\-to]+\s*(\d+(?:\.\d+)?)\s*([a-zA-Zµ°/³·]+)"
)


def _extract_specs_from_procedure(proc: dict) -> list[dict]:
    """
    Scan a normalised procedure's step instructions, expected_results, symptoms,
    and causes for numeric specifications.

    Returns a list of SpecRecord-compatible dicts (embedding field is empty —
    embeddings are added in batch at save time by _save_spec_database()).

    Design:
      • One SpecRecord per unique (component, spec_type, value) triple.
      • acceptable_range is populated when a range pattern is found in the
        same text field as the value (e.g. "23.75–26.25µF").
      • failure_if_out_of_range is populated from the procedure's symptoms.
      • repair_actions is populated from the first 3 step instructions.
    """
    specs:     list[dict] = []
    component  = proc.get("component", "Unknown")
    manual_src = proc.get("manual_source", "")
    section    = proc.get("source_section", "")

    # Collect all searchable text with a field label for debugging
    search_texts: list[tuple[str, str]] = []
    for step in proc.get("step_sequence", []):
        search_texts.append(("step_instruction",  step.get("instruction",     "")))
        search_texts.append(("step_expected",      step.get("expected_result", "")))
    for sym in proc.get("symptoms", []):
        search_texts.append(("symptom", sym))
    for cause in proc.get("causes", []):
        search_texts.append(("cause", cause))

    seen_specs: set[str] = set()

    for _field_label, text in search_texts:
        if not text:
            continue
        for spec_type, pattern, default_unit in _SPEC_PATTERNS:
            for m in re.finditer(pattern, text, re.I):
                value = m.group(1)
                key   = f"{component.lower()}::{spec_type}::{value}"
                if key in seen_specs:
                    continue
                seen_specs.add(key)

                # Try to find a numeric range near this value
                acceptable_range = ""
                range_m = _RANGE_RE.search(text)
                if range_m:
                    r_lo, r_hi, r_unit = range_m.groups()
                    acceptable_range = f"{r_lo}–{r_hi} {r_unit.strip()}"

                failure_symptoms = proc.get("symptoms", [])[:3]
                repair_actions   = [
                    s.get("instruction", "")
                    for s in proc.get("step_sequence", [])[:3]
                    if s.get("instruction")
                ]

                specs.append({
                    "component":               component,
                    "spec_type":               spec_type,
                    "value":                   value,
                    "unit":                    default_unit,
                    "acceptable_range":        acceptable_range,
                    "source_page":             section,
                    "manual_source":           manual_src,
                    "failure_if_out_of_range": failure_symptoms,
                    "repair_actions":          repair_actions,
                    "embedding":               [],   # populated at save time
                })

    return specs


# ── Component graph builder ───────────────────────────────────────────────────
# Maintains an in-memory graph during batch processing.
# _resolve_component_alias() does the cross-PDF canonicalisation.
# _update_component_graph() merges one procedure into the graph.
# _save_component_graph() serialises to JSON at end of batch.

def _update_component_graph(
    graph:         dict[str, dict],
    proc:          dict,
    image_records: list[dict] | None = None,
) -> None:
    """
    Merge one procedure into the in-memory component graph.

    Args:
        graph:         shared dict keyed by canonical component name (lowercase).
                       Mutated in-place. Accumulated across all PDFs in a batch.
        proc:          normalised procedure dict.
        image_records: captioned image records from the same PDF.
                       Used to link diagram_sources to component nodes.

    Alias resolution:
        _resolve_component_alias() uses _component_embeddings (module-level cache)
        for O(N) cosine comparison with a single np.dot per existing key.
        If similarity ≥ _ALIAS_SIMILARITY_THRESHOLD the incoming name is registered
        as an alias of the existing canonical node rather than creating a duplicate.
    """
    raw_component = proc.get("component", "").strip()
    if not raw_component or raw_component.lower() in ("unknown", ""):
        return

    canonical_key = _resolve_component_alias(raw_component)

    # ── Initialise node if new ────────────────────────────────────────────────
    if canonical_key not in graph:
        graph[canonical_key] = {
            "component":       raw_component,
            "aliases":         [],
            "connected_to":    [],
            "function":        "",   # what this component does
            "diagram_sources": [],
            "manual_sources":  [],
            "related_faults":  [],
            "related_specs":   [],
            "machine_family":  proc.get("machine_family", ""),
            "embedding_text":  "",   # populated at save time
        }

    node = graph[canonical_key]

    # Register alias if the raw name differs from the canonical form
    raw_lower = raw_component.lower()
    if raw_lower != canonical_key and raw_lower not in node["aliases"]:
        node["aliases"].append(raw_lower)

    # ── Merge connected_to ────────────────────────────────────────────────────
    for connected_part in proc.get("connected_to", []):
        cp = connected_part.lower().strip()
        if cp and cp not in node["connected_to"]:
            node["connected_to"].append(cp)

    # ── Merge if_wrong_installation into related_faults ───────────────────────
    for wrong_sym in proc.get("if_wrong_installation", []):
        if wrong_sym and wrong_sym not in node["related_faults"]:
            node["related_faults"].append(wrong_sym)

    # ── Merge manual source ───────────────────────────────────────────────────
    src = proc.get("manual_source", "")
    if src and src not in node["manual_sources"]:
        node["manual_sources"].append(src)

    # ── Merge fault symptoms ──────────────────────────────────────────────────
    for sym in proc.get("symptoms", [])[:5]:
        if sym and sym not in node["related_faults"]:
            node["related_faults"].append(sym)

    # ── Merge spec values from causal procedures ──────────────────────────────
    if proc.get("knowledge_type") == "causal_inferred":
        for step in proc.get("step_sequence", []):
            expected = step.get("expected_result", "").strip()
            if expected and len(expected) > 4 and expected not in node["related_specs"]:
                node["related_specs"].append(expected)

    # ── Link image diagram sources ────────────────────────────────────────────
    if image_records:
        comp_lower = raw_component.lower()
        for img_rec in image_records:
            shown = [c.lower() for c in img_rec.get("components_shown", [])]
            if comp_lower in shown or any(comp_lower in c for c in shown):
                fname = img_rec.get("filename", "")
                if fname and fname not in node["diagram_sources"]:
                    node["diagram_sources"].append(fname)

    # ── Merge relationship edges from image captions ──────────────────────────
    # caption_images_with_gemini() now returns connected_relationships lists;
    # fold them into the graph so the component graph reflects image knowledge too.
    if image_records:
        for img_rec in image_records:
            for rel in img_rec.get("connected_relationships", []):
                part_from = rel.get("from", "").lower().strip()
                part_to   = rel.get("to",   "").lower().strip()
                # If THIS node is one of the endpoints, link the other end
                if part_from == canonical_key and part_to:
                    if part_to not in node["connected_to"]:
                        node["connected_to"].append(part_to)
                elif part_to == canonical_key and part_from:
                    if part_from not in node["connected_to"]:
                        node["connected_to"].append(part_from)


# ── Auxiliary database save helpers ──────────────────────────────────────────
#
# Embedding policy (applied in ALL save helpers):
#   JSON files store only embedding_text (human-readable string).
#   Raw float vectors are NOT saved in JSON — they are computed separately
#   and loaded into a vector store (Chroma / FAISS) at index time.
#   This keeps JSON human-readable and avoids ~4 KB of floats per record.

def _save_component_graph(graph: dict[str, dict], output_path: Path) -> None:
    """
    Build embedding_text per node and serialise the component graph.
    Embedding input: component name + aliases + connected_to + function.
    Vectors are NOT saved in JSON — only embedding_text.
    """
    if not graph:
        logger.warning("⚠️  Component graph is empty — skipping save.")
        return

    nodes = list(graph.values())

    for node in nodes:
        parts = (
            [node["component"]]
            + node.get("aliases", [])[:3]
            + node.get("connected_to", [])[:4]
        )
        func = node.get("function", "")
        if func:
            parts.append(func)
        faults = node.get("related_faults", [])[:3]
        parts.extend(faults)
        node["embedding_text"] = " ".join(p for p in parts if p)
        # Remove raw vector if present from old runs
        node.pop("embedding", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)
    logger.info("💾 Component graph saved: %d nodes → %s", len(nodes), output_path)


def _save_spec_database(specs: list[dict], output_path: Path) -> None:
    """
    Build embedding_text per spec record and save.
    Embedding input: component + parameter + value + unit + if_out_of_range.
    Vectors NOT saved in JSON.
    """
    if not specs:
        logger.info("ℹ️  No specs extracted — spec_database.json not written.")
        return

    for s in specs:
        param  = s.get("parameter") or s.get("spec_type", "")
        oor    = " ".join((s.get("if_out_of_range") or s.get("failure_if_out_of_range") or [])[:2])
        s["embedding_text"] = f"{s['component']} {param} {s['value']} {s['unit']} {oor}".strip()
        # Sync field aliases
        s["parameter"]  = param
        s["page"]       = s.get("page") or s.get("source_page", "")
        s["if_out_of_range"] = s.get("if_out_of_range") or s.get("failure_if_out_of_range") or []
        # Remove raw vector if present from old runs
        s.pop("embedding", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2, ensure_ascii=False)
    logger.info("💾 Spec database saved: %d records → %s", len(specs), output_path)


def _save_image_knowledge_db(images: list[dict], output_path: Path) -> None:
    """
    Build embedding_text per image record and save.
    Embedding input: caption + diagram_type + components + fault_relevance + keywords.
    Vectors NOT saved in JSON.
    """
    if not images:
        logger.info("ℹ️  No images — image knowledge DB not written.")
        return

    for img in images:
        caption    = img.get("caption", "")
        diagram    = img.get("diagram_type_classifier", "")
        components = " ".join(img.get("components_visible") or img.get("components_shown", []))
        fault_rel  = " ".join(img.get("fault_relevance") if isinstance(img.get("fault_relevance"), list)
                              else ([img.get("fault_relevance", "")] if img.get("fault_relevance") else []))
        keywords   = " ".join(img.get("search_keywords", []))
        repair_rel = " ".join(img.get("repair_relevance", []))
        img["embedding_text"] = " ".join(filter(None, [
            caption, diagram, components, fault_rel, repair_rel, keywords,
        ]))
        # Remove raw vector if present from old runs
        img.pop("embedding", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)
    logger.info("💾 Image knowledge DB saved: %d records → %s", len(images), output_path)


def _save_fault_library(faults: list[dict], output_path: Path) -> None:
    """
    Save the fault library.
    embedding_text: symptom + likely_causes + understanding + verify.
    Vectors NOT saved in JSON.
    """
    if not faults:
        logger.info("ℹ️  No fault entries — fault_library.json not written.")
        return

    for f in faults:
        causes     = " ".join(f.get("likely_causes", [])[:3])
        verify     = " ".join(f.get("verify", [])[:3])
        understand = f.get("understanding", "")
        f["embedding_text"] = " ".join(filter(None, [
            f.get("symptom", ""), f.get("component", ""),
            causes, understand, verify,
        ]))

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(faults, f_out, indent=2, ensure_ascii=False)
    logger.info("💾 Fault library saved: %d entries → %s", len(faults), output_path)


def _save_diagnostic_trees(trees: list[dict], output_path: Path) -> None:
    """
    Save the diagnostic tree library.
    embedding_text: symptom + component + all branch questions.
    Vectors NOT saved in JSON.
    """
    if not trees:
        logger.info("ℹ️  No diagnostic trees — diagnostic_trees.json not written.")
        return

    for t in trees:
        questions = " ".join(
            b.get("question", "") for b in t.get("branches", [])
        )
        t["embedding_text"] = " ".join(filter(None, [
            t.get("symptom", ""), t.get("component", ""), questions,
        ]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trees, f, indent=2, ensure_ascii=False)
    logger.info("💾 Diagnostic trees saved: %d trees → %s", len(trees), output_path)


def _save_repair_procedures(procedures: list[dict], output_path: Path) -> None:
    """
    Save named repair procedures.
    embedding_text: procedure name + component + steps.
    Vectors NOT saved in JSON.
    """
    if not procedures:
        logger.info("ℹ️  No repair procedures — repair_procedures.json not written.")
        return

    for p in procedures:
        steps_text = " ".join(p.get("steps", [])[:5])
        p["embedding_text"] = " ".join(filter(None, [
            p.get("procedure", ""), p.get("component", ""), steps_text,
        ]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(procedures, f, indent=2, ensure_ascii=False)
    logger.info("💾 Repair procedures saved: %d → %s", len(procedures), output_path)


def _build_and_save_knowledge_index(
    procedures:      list[dict],
    specs:           list[dict],
    images:          list[dict],
    faults:          list[dict],
    component_graph: dict[str, dict],
    output_path:     Path,
) -> None:
    """
    Build the cross-link knowledge index and save to knowledge_index.json.

    One entry per canonical component, cross-linking:
      procedure_ids  → chunk_ids from procedures DB
      fault_ids      → symptom strings from fault library
      spec_ids       → "spec_type:component" keys from spec DB
      image_ids      → filenames from image DB
      connected_components → from component graph
      embedding_text → human-readable; vectors computed separately (NOT in JSON)
    """
    index: dict[str, dict] = {}

    def _get_or_create(comp_raw: str, comp_display: str) -> dict:
        key = comp_raw.lower().strip()
        if not key:
            return {}
        if key not in index:
            index[key] = {
                "component":            comp_display or comp_raw,
                "procedure_ids":        [],
                "fault_ids":            [],
                "spec_ids":             [],
                "image_ids":            [],
                "connected_components": [],
                "embedding_text":       "",
            }
        return index[key]

    # Populate from procedures
    for proc in procedures:
        entry = _get_or_create(proc.get("component", ""), proc.get("component", ""))
        if not entry:
            continue
        cid = proc.get("chunk_id", "")
        if cid and cid not in entry["procedure_ids"]:
            entry["procedure_ids"].append(cid)

    # Populate from fault library
    for fault in faults:
        entry = _get_or_create(fault.get("component", ""), fault.get("component", ""))
        if not entry:
            continue
        sym = fault.get("symptom", "")
        if sym and sym not in entry["fault_ids"]:
            entry["fault_ids"].append(sym)

    # Cross-link specs
    for spec in specs:
        comp = spec.get("component", "").lower().strip()
        if comp and comp in index:
            param = spec.get("parameter") or spec.get("spec_type", "")
            sid = f"{param}:{comp}"
            if sid not in index[comp]["spec_ids"]:
                index[comp]["spec_ids"].append(sid)

    # Cross-link images
    for img in images:
        shown = img.get("components_visible") or img.get("components_shown", [])
        for comp_name in shown:
            comp = comp_name.lower().strip()
            if comp in index:
                fname = img.get("filename", "")
                if fname and fname not in index[comp]["image_ids"]:
                    index[comp]["image_ids"].append(fname)

    # Pull connected_components and build embedding_text from graph
    for comp_key, entry in index.items():
        node = component_graph.get(comp_key, {})
        entry["connected_components"] = node.get("connected_to", [])
        # Build embedding_text from component name + connections + faults
        parts = [entry["component"]] + entry["connected_components"][:4]
        parts += entry["fault_ids"][:3]
        entry["embedding_text"] = " ".join(p for p in parts if p)

    entries = list(index.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    logger.info("💾 Knowledge index saved: %d entries → %s", len(entries), output_path)


# ── Deduplication ─────────────────────────────────────────────────────────────
#
# A procedure is a duplicate if it shares the same (system, component, symptoms)
# fingerprint as one already in the database — regardless of which manual it
# came from. Fingerprint is an MD5 of the sorted, lowercased tuple so minor
# wording differences don't create false duplicates.

def _fingerprint(proc: dict) -> str:
    import hashlib as _hl
    system    = proc.get("system", "").lower().strip()
    component = proc.get("component", "").lower().strip()
    # Prefer canonical symptoms so "motor won't start" and "fails to start"
    # hash to the same fingerprint and are correctly deduplicated.
    symptom_source = proc.get("symptoms_canonical") or proc.get("symptoms", [])
    symptoms  = "|".join(sorted(s.lower().strip() for s in symptom_source))
    raw       = f"{system}::{component}::{symptoms}"
    return _hl.md5(raw.encode()).hexdigest()


def deduplicate(procedures: list[dict]) -> list[dict]:
    """
    Remove duplicate procedures by (system, component, symptoms) fingerprint.
    First occurrence wins; later duplicates are logged and dropped.
    """
    seen:   set[str]   = set()
    unique: list[dict] = []
    dupes = 0
    for proc in procedures:
        fp = _fingerprint(proc)
        if fp in seen:
            dupes += 1
        else:
            seen.add(fp)
            unique.append(proc)
    if dupes:
        logger.info("   🔁 Deduplication: %d duplicate(s) removed, %d unique kept", dupes, len(unique))
    return unique


# ── FIX #2: Image extraction ─────────────────────────────────────────────────
# Agricultural service manuals are 60–80% visual. Wiring diagrams, exploded-view
# drawings, impeller clearance tables, and assembly photos were ALL being discarded.
# This adds an image pass to every PDF processed.

import base64 as _base64

# ── Diagram type taxonomy ─────────────────────────────────────────────────────
#
# Gemini Vision returns free-text diagram_type values like "exploded view",
# "cross section diagram", "wiring", etc.  Without normalisation those values
# become useless for ChromaDB metadata filtering (every string is unique).
#
# _DIAGRAM_TAXONOMY maps every plausible Gemini synonym to one canonical label.
# _classify_diagram_type() applies the map via substring matching so partial
# or combined strings ("exploded assembly diagram") still resolve correctly.
# The canonical value is stored as diagram_type_classifier alongside the raw
# Gemini string (diagram_type) so nothing is lost and both are queryable.
#
# Canonical labels (stable — add aliases freely, never rename these):
#   exploded_view     — disassembled part layout showing component positions
#   wiring_diagram    — electrical circuit / wiring harness schematic
#   hydraulic_layout  — hydraulic circuit / fluid power schematic
#   sectional_view    — cross-section / cut-through showing internal geometry
#   assembly_sequence — numbered step-by-step assembly or installation drawings
#   flow_chart        — decision / diagnostic / process flow diagram
#   torque_table      — table of torque values, clearances, or specifications
#   dimension_drawing — outline / envelope / mounting-hole dimension drawing
#   photo             — photographic image of the actual part or machine
#   chart             — performance curve, graph, or data chart
#   other             — fallback for anything that doesn't match

_DIAGRAM_TAXONOMY: dict[str, str] = {
    # exploded_view
    "exploded":         "exploded_view",
    "disassembl":       "exploded_view",
    "part layout":      "exploded_view",
    "component layout": "exploded_view",
    "breakdown view":   "exploded_view",
    # wiring_diagram
    "wiring":           "wiring_diagram",
    "electrical schematic": "wiring_diagram",
    "circuit diagram":  "wiring_diagram",
    "schematic":        "wiring_diagram",
    "harness":          "wiring_diagram",
    # hydraulic_layout
    "hydraulic":        "hydraulic_layout",
    "fluid circuit":    "hydraulic_layout",
    "pneumatic":        "hydraulic_layout",
    # sectional_view
    "cross.section":    "sectional_view",     # matches "cross-section" and "cross section"
    "section view":     "sectional_view",
    "sectional":        "sectional_view",
    "cut.through":      "sectional_view",
    "cut away":         "sectional_view",
    "cutaway":          "sectional_view",
    # assembly_sequence
    "assembly sequence": "assembly_sequence",
    "installation step": "assembly_sequence",
    "assembly step":    "assembly_sequence",
    "fitment":          "assembly_sequence",
    "assembly procedure": "assembly_sequence",
    # flow_chart
    "flowchart":        "flow_chart",
    "flow chart":       "flow_chart",
    "flow diagram":     "flow_chart",
    "decision":         "flow_chart",
    "diagnostic chart": "flow_chart",
    "troubleshoot":     "flow_chart",
    # torque_table
    "torque":           "torque_table",
    "torque table":     "torque_table",
    "specification table": "torque_table",
    "clearance table":  "torque_table",
    "tolerance table":  "torque_table",
    "spec table":       "torque_table",
    "tightening":       "torque_table",
    # dimension_drawing
    "dimension":        "dimension_drawing",
    "outline drawing":  "dimension_drawing",
    "mounting":         "dimension_drawing",
    "envelope":         "dimension_drawing",
    # photo
    "photo":            "photo",
    "photograph":       "photo",
    "actual":           "photo",
    # chart
    "performance curve": "chart",
    "graph":            "chart",
    "chart":            "chart",
    "curve":            "chart",
}

# Ordered list of (pattern_substring, canonical) for fast linear scan.
# More-specific strings must come before shorter ones that would shadow them.
_TAXONOMY_PAIRS: list[tuple[str, str]] = sorted(
    _DIAGRAM_TAXONOMY.items(),
    key=lambda kv: len(kv[0]),
    reverse=True,           # longest (most specific) first
)

_CANONICAL_TYPES: frozenset[str] = frozenset({
    "exploded_view", "wiring_diagram", "hydraulic_layout", "sectional_view",
    "assembly_sequence", "flow_chart", "torque_table", "dimension_drawing",
    "photo", "chart", "other",
})


def _classify_diagram_type(raw_gemini_value: str) -> str:
    """
    Map a free-text Gemini diagram_type string to a canonical label.

    Strategy:
      1. If Gemini already returned a canonical label → use it directly.
      2. Scan _TAXONOMY_PAIRS (longest substring first) for a match.
      3. Fall back to "other".

    The function is pure and deterministic — same input always gives same output.
    """
    if not raw_gemini_value:
        return "other"

    lowered = raw_gemini_value.lower().strip()

    # Exact canonical match (Gemini occasionally returns perfect values)
    canonical_lowered = lowered.replace(" ", "_").replace("-", "_")
    if canonical_lowered in _CANONICAL_TYPES:
        return canonical_lowered

    # Substring scan (handles "exploded assembly diagram", "cross-section view", etc.)
    for pattern, canonical in _TAXONOMY_PAIRS:
        # Use regex so "cross.section" matches both "cross-section" and "cross section"
        if re.search(pattern, lowered):
            return canonical

    return "other"


# Regex that matches figure/table references commonly found in equipment manuals.
# Covers: "Fig 5", "Fig. 5", "Figure 5", "Figure 5.2", "Fig5", "Table 3A",
#          "Table 3-A", "Ref. B-22", "See Fig. 12" etc.
_FIG_REF_RE = re.compile(
    r"\b((?:Fig(?:ure)?\.?\s*\d+[\w.-]*)|(?:Table\s+\d+[\w.-]*))\b",
    re.IGNORECASE,
)

# ── Image pipeline tuning constants ──────────────────────────────────────────
# Adjust these to trade off coverage vs. speed / API cost.

# Minimum image pixel size. 450 px eliminates logos, arrows, tiny symbols.
_MIN_IMAGE_PX = 450

# Minimum raw image file size. < 30 KB = icon / stamp / blank whitespace.
_MIN_IMAGE_BYTES = 30_720      # 30 KB

# Final cap AFTER scoring. Smart scoring selects the best N, not the first N.
_MAX_IMAGES_PER_PDF = 15

# Maximum surrounding page text stored in the DB record (chars).
_MAX_SURROUNDING_TEXT = 1_500

# Surrounding text cap inside the Gemini prompt (tighter = cheaper).
_PROMPT_SURROUNDING_TEXT = 700

# Neighbour context strings fed to Gemini.
_PROMPT_NEIGHBOUR_CTX = 150

# Parallel workers for Gemini Vision. 3 is safe on the free tier.
_CAPTION_WORKERS = 1

# ── Pre-Gemini filter sets ────────────────────────────────────────────────────

_PAGE_SKIP_TERMS: frozenset[str] = frozenset({
    "table of contents", "contents", "index", 
    "blank page", "this page intentionally left blank"
})

_SECTION_SKIP_RE = re.compile(
    r"\b(warranty|foreword|contents?|index|abbreviations?)\b",
    re.IGNORECASE,
)

# Logo / branding detection: if ALL of these conditions are true the image
# is a brand logo and should be skipped:
#   • image is squarish (aspect ratio close to 1:1)
#   • image area < 500 × 500 px
#   • page text contains a known brand/logo keyword
_LOGO_PAGE_KEYWORDS: frozenset[str] = frozenset({
    "logo", "brand", "enriching lives",
    "kirloskar", "grundfos", "crompton", "ksb", "lubi", "texmo",
    "wilo", "flowserve", "sulzer", "ebara",
    "registered trademark", "all rights reserved",
})

# ── Scoring weights ───────────────────────────────────────────────────────────
# Each image is scored BEFORE the cap is applied. The top-N highest-scoring
# images are sent to Gemini. This is far smarter than "keep the first N".

_SCORE_LARGE_IMAGE      =  2    # width × height > 500 000 px²
_SCORE_FIGURE_REF       =  3    # page text has an explicit "Fig X" / "Table Y"
_SCORE_DIAGNOSTIC_SEC   =  4    # section is a known high-value diagnostic section
_SCORE_MECHANICAL_TERMS =  3    # page text contains mechanical/fault keywords
_SCORE_LOGO_PENALTY     = -10   # logo/brand image detected
_SCORE_WARRANTY_PENALTY = -10   # warranty/legal page detected (belt-and-suspenders)

# Keywords that indicate a HIGH-VALUE diagnostic page/section
_DIAGNOSTIC_TERMS: frozenset[str] = frozenset({
    "troubleshoot", "fault", "failure", "maintenance", "overhaul",
    "dismantl", "assemble", "replace", "repair", "bearing", "seal",
    "impeller", "torque", "clearance", "alignment", "wiring", "hydraulic",
    "leakage", "vibration", "noise", "cavitation", "priming",
})


def _score_image(
    width: int,
    height: int,
    raw_page_text: str,
    figure_references: list[str],
    section_ctx: str,
) -> int:
    """
    Assign a diagnostic-value score to a candidate image.
    Higher = more worth sending to Gemini.
    Called before writing the image file to disk.
    """
    score = 0
    page_lower = raw_page_text.lower()

    # Large images are more likely to be real diagrams
    if width * height > 500_000:
        score += _SCORE_LARGE_IMAGE

    # Explicit figure references on the page = diagram is labelled and referenced
    if figure_references:
        score += _SCORE_FIGURE_REF

    # Page or section contains maintenance/fault language
    if any(term in page_lower for term in _DIAGNOSTIC_TERMS):
        score += _SCORE_MECHANICAL_TERMS

    # Section heading is a known high-value section
    section_lower = section_ctx.lower()
    if any(t in section_lower for t in _DIAGNOSTIC_TERMS):
        score += _SCORE_DIAGNOSTIC_SEC

    # Logo/branding penalty
    if (width * height < 500 * 500 and
            any(k in page_lower for k in _LOGO_PAGE_KEYWORDS)):
        score += _SCORE_LOGO_PENALTY

    # Warranty/legal page penalty (belt-and-suspenders — page-level skip already
    # catches most, but some pages are mislabelled)
    if any(t in page_lower for t in ("warranty", "copyright", "all rights reserved")):
        score += _SCORE_WARRANTY_PENALTY

    return score


def extract_images_from_pdf(
    pdf_path: Path,
    output_dir: Path,
    page_section_map: dict[int, str] | None = None,
) -> list[dict]:
    """
    Extract and PRE-FILTER embedded images from a PDF before any Gemini call.

    Filter pipeline (cheap → expensive, all local, zero API cost):
      1. Page-level text skip  — entire page skipped if it contains warranty /
                                 contents / safety boilerplate (_PAGE_SKIP_TERMS)
      2. Section-level skip    — skip if section heading matches low-value pattern
      3. Pixel size filter     — skip if width or height < _MIN_IMAGE_PX (450 px)
      4. Byte size filter      — skip if raw bytes < _MIN_IMAGE_BYTES (30 KB)
      5. Logo detection        — skip squarish small images on branded pages
      6. Perceptual hash dedup — skip visually identical images (logos repeat
                                 across every page header/footer)
      7. Priority scoring      — score every surviving candidate; keep only the
                                 top _MAX_IMAGES_PER_PDF by score rather than
                                 by page order

    Attaches: section_context, surrounding_text, figure_references,
              prev_context, next_context, priority_score.
    """
    import fitz
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))

    if page_section_map is None:
        page_section_map = {}

    # ── Perceptual hash deduplication setup ───────────────────────────────────
    # imagehash (pip install imagehash pillow) gives fast perceptual hashing.
    # Falls back to MD5 of raw bytes if imagehash is not installed — less
    # accurate (won't catch JPEG vs PNG of the same logo) but still useful.
    try:
        import imagehash as _imagehash
        from PIL import Image as _PILImage
        import io as _io
        _USE_PHASH = True
    except ImportError:
        _USE_PHASH = False
        logger.debug("imagehash not installed — using MD5 dedup (pip install imagehash pillow for better dedup)")

    seen_hashes: set[str] = set()

    # Counters for the filter summary log line
    sk_page_text = sk_section = sk_px = sk_bytes = sk_logo = sk_hash = 0

    # Collect ALL candidates with their scores first, then rank and cap.
    # This is the key architectural change: "score-then-cap" vs "first-N".
    candidates: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        section_ctx = page_section_map.get(page_num, "General")

        # ── Filter 1: Section-level skip ──────────────────────────────────────
        if _SECTION_SKIP_RE.search(section_ctx):
            sk_section += len(page.get_images(full=True))
            continue

        # ── FIX P1: Page text (read once, shared across all images on page) ───
        raw_page_text    = page.get_text("text")
        page_lower       = raw_page_text.lower()
        surrounding_text = re.sub(r"\n{3,}", "\n\n", raw_page_text).strip()
        surrounding_text = surrounding_text[:_MAX_SURROUNDING_TEXT]

        # ── Filter 2: Page-level text skip ────────────────────────────────────
        # Checks the ACTUAL OCR text, not just the section header — catches
        # cover pages, warranty pages, and other boilerplate that may not have
        # a heading at all (they often don't).
        if any(term in page_lower for term in _PAGE_SKIP_TERMS):
            sk_page_text += len(page.get_images(full=True))
            continue

        # ── FIX P2: Figure references ──────────────────────────────────────────
        figure_references: list[str] = list(
            dict.fromkeys(
                m.group(1).strip()
                for m in _FIG_REF_RE.finditer(raw_page_text)
            )
        )

        for img_idx, img in enumerate(page.get_images(full=True)):
            xref       = img[0]
            base_image = doc.extract_image(xref)
            width      = base_image.get("width",  0)
            height     = base_image.get("height", 0)

            # ── Filter 3: Pixel size ───────────────────────────────────────────
            if width < _MIN_IMAGE_PX or height < _MIN_IMAGE_PX:
                sk_px += 1
                continue

            img_data = base_image["image"]

            # ── Filter 4: Byte size ────────────────────────────────────────────
            if len(img_data) < _MIN_IMAGE_BYTES:
                sk_bytes += 1
                continue

            # ── Filter 5: Logo detection ───────────────────────────────────────
            # Logo heuristic: small + near-square + branded page text.
            # Aspect ratio between 0.5 and 2.0 = "squarish".
            aspect = width / height if height else 0
            is_small    = width * height < 500 * 500
            is_squarish = 0.5 <= aspect <= 2.0
            has_brand   = any(k in page_lower for k in _LOGO_PAGE_KEYWORDS)
            if is_small and is_squarish and has_brand:
                sk_logo += 1
                continue

            # ── Filter 6: Perceptual hash deduplication ────────────────────────
            if _USE_PHASH:
                try:
                    pil_img  = _PILImage.open(_io.BytesIO(img_data))
                    img_hash = str(_imagehash.phash(pil_img))
                except Exception:
                    # Corrupt / undecodable image — use byte hash as fallback
                    import hashlib
                    img_hash = hashlib.md5(img_data).hexdigest()
            else:
                import hashlib
                img_hash = hashlib.md5(img_data).hexdigest()

            if img_hash in seen_hashes:
                sk_hash += 1
                continue
            seen_hashes.add(img_hash)

            # ── Score this candidate ────────────────────────────────────────────
            score = _score_image(width, height, raw_page_text,
                                 figure_references, section_ctx)

            candidates.append({
                "filename":                "",          # filled after ranking
                "page":                    page_num,
                "width":                   width,
                "height":                  height,
                "manual_source":           pdf_path.name,
                "section_context":         section_ctx,
                "surrounding_text":        surrounding_text,
                "figure_references":       figure_references,
                "caption":                 "",
                "diagram_type":            "",
                "diagram_type_classifier": "other",
                "components_shown":        [],
                "fault_relevance":         "",
                "search_keywords":         [],
                "prev_context":            "",
                "next_context":            "",
                "priority_score":          score,
                # Temporary storage — img_data written to disk only for kept images
                "_img_data":               img_data,
                "_ext":                    base_image["ext"],
                "_img_idx":                img_idx,
            })

    doc.close()

    # ── Filter 7: Score-based ranking and cap ─────────────────────────────────
    # Sort by descending score, then by page number as a tiebreaker (earlier
    # pages in a section tend to have the most important diagrams).
    candidates.sort(key=lambda r: (-r["priority_score"], r["page"]))
    kept = candidates[:_MAX_IMAGES_PER_PDF]

    # Write only the kept images to disk, then clean up temp fields
    image_records: list[dict] = []
    for rec in kept:
        img_data = rec.pop("_img_data")
        ext      = rec.pop("_ext")
        img_idx  = rec.pop("_img_idx")
        filename = f"{pdf_path.stem}_p{rec['page']}_img{img_idx}.{ext}"
        (output_dir / filename).write_bytes(img_data)
        rec["filename"] = filename
        image_records.append(rec)

    # Sort kept records back into page order for neighbour linking
    image_records.sort(key=lambda r: (r["page"], r["filename"]))

    logger.info(
        "   🖼️  %s: %d/%d candidates kept (cap=%d) | "
        "skipped: %d page-text, %d section, %d too-small-px, "
        "%d too-small-bytes, %d logos, %d duplicates",
        pdf_path.name, len(image_records), len(candidates) + sk_page_text +
        sk_section + sk_px + sk_bytes + sk_logo + sk_hash,
        _MAX_IMAGES_PER_PDF,
        sk_page_text, sk_section, sk_px, sk_bytes, sk_logo, sk_hash,
    )

    # FIX D: Fill physical neighbour context
    for i, rec in enumerate(image_records):
        if i > 0:
            prev = image_records[i - 1]
            rec["prev_context"] = (
                f"{prev['filename']} (p{prev['page']}, "
                f"section: {prev['section_context']})"
            )
        if i < len(image_records) - 1:
            nxt = image_records[i + 1]
            rec["next_context"] = (
                f"{nxt['filename']} (p{nxt['page']}, "
                f"section: {nxt['section_context']})"
            )

    return image_records


def _preprocess_image_for_ocr(img_bytes: bytes, ext: str) -> tuple[bytes, str]:
    """
    Fix 7: Preprocess a raw image before sending to Gemini Vision.

    Pipeline (all local, zero API cost):
      1. Convert to greyscale  — removes colour noise; helps Gemini read faint
         pencil-drawn arrows and low-contrast text common in scanned pump manuals.
      2. Sharpen               — UnsharpMask recovers soft edges from JPEG
         compression / photocopy scanning.
      3. 2× upscale            — doubles pixel dimensions via LANCZOS resampling
         so small 450–600 px diagrams reach ≥ 900 px; Gemini Vision accuracy
         improves noticeably above ~800 px on text-heavy diagrams.

    Only applied when Pillow is available (it's already a pymupdf4llm dependency).
    Falls back silently to the original bytes if Pillow is missing or if the
    image format is unsupported (e.g. JBIG2 from old scanned PDFs).

    Returns: (processed_bytes, mime_type) — always valid even on error.
    """
    try:
        from PIL import Image as _PILImage, ImageFilter as _ImageFilter
        import io as _io

        pil = _PILImage.open(_io.BytesIO(img_bytes))

        # 1. Greyscale
        pil = pil.convert("L").convert("RGB")   # L→RGB so JPEG output is valid

        # 2. Sharpen (UnsharpMask is gentler than SHARPEN kernel — avoids haloing)
        pil = pil.filter(_ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        # 3. 2× upscale only if image is small (≤ 800 px on either side)
        w, h = pil.size
        if w <= 800 or h <= 800:
            pil = pil.resize((w * 2, h * 2), _PILImage.LANCZOS)

        buf = _io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"

    except Exception:
        # Pillow missing or image unreadable — pass original bytes through
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        return img_bytes, mime


def _caption_single_image(args: tuple) -> dict:
    """
    Caption one image record via Gemini Vision.
    Designed to be called from a ThreadPoolExecutor worker.

    Returns the record dict with Gemini fields merged in (mutates in-place
    AND returns for executor.map compatibility).

    Fix 4: AFC disabled — removes ~1-2 s overhead per call.
    Fix 5: Prompt text caps applied (_PROMPT_SURROUNDING_TEXT, _PROMPT_NEIGHBOUR_CTX).
    Fix 7: Image preprocessed (greyscale + sharpen + 2× upscale) before Gemini.
    """
    record, image_dir = args
    img_path = image_dir / record["filename"]
    import time as _time
    _start = _time.time()
    logger.info("   📤 Uploading %s (%.1f KB)...", record["filename"], img_path.stat().st_size / 1024)

    try:
        img_bytes = img_path.read_bytes()
        ext       = img_path.suffix.lstrip(".").lower()

        # Fix 7: preprocess for OCR quality (greyscale → sharpen → 2× upscale)
        img_bytes, mime = _preprocess_image_for_ocr(img_bytes, ext)
        b64 = _base64.standard_b64encode(img_bytes).decode()

        section_ctx      = record.get("section_context", "General")
        surrounding_text = record.get("surrounding_text", "").strip()
        figure_refs      = record.get("figure_references", [])
        prev_ctx         = record.get("prev_context", "")[:_PROMPT_NEIGHBOUR_CTX]
        next_ctx         = record.get("next_context", "")[:_PROMPT_NEIGHBOUR_CTX]

        # ── Context blocks (rendered only when non-empty) ─────────────────────
        neighbour_lines = ""
        if prev_ctx:
            neighbour_lines += f"Prev image: {prev_ctx}\n"
        if next_ctx:
            neighbour_lines += f"Next image: {next_ctx}\n"

        fig_ref_line = ""
        if figure_refs:
            fig_ref_line = "Fig/table refs on page: " + ", ".join(figure_refs) + "\n"

        # Fix 5: cap surrounding text at _PROMPT_SURROUNDING_TEXT (700 chars)
        surrounding_block = ""
        if surrounding_text:
            surrounding_block = (
                "\n--- PAGE TEXT ---\n"
                + surrounding_text[:_PROMPT_SURROUNDING_TEXT]
                + "\n--- END ---\n"
            )

        prompt = (
            "You are reading a page from an agricultural equipment service manual.\n"
            "Machine family: agricultural machinery and equipment "
            "(Tractors, Power Tillers, Harvesters, Water Pumps, Electric Motors, etc.).\n"
            "(treat as one connected family — parts are shared across these machines).\n"
            f"Section: '{section_ctx}'.\n"
            f"{fig_ref_line}"
            f"{neighbour_lines}"
            f"{surrounding_block}\n"
            "Analyse this image as a field repair technician, not as a document librarian.\n"
            "Your goal: extract everything that helps diagnose faults and guide repair.\n\n"
            "EXTRACTION RULES:\n"
            "1. ARROWS: Every arrow pointing to a part must become a relationship entry.\n"
            "   Extract label text + part the arrow points to.\n"
            "2. LABELS/CALLOUTS: Extract every number, letter, or symbol callout.\n"
            "3. EXPLODED/ASSEMBLY VIEWS: Identify which part attaches to which.\n"
            "   relationship type examples: mounted_on, bolts_to, seals_against,\n"
            "   connects_to, drives, surrounds, inserted_into, aligned_with.\n"
            "4. SPEC TABLES: Every row → extract as 'param: value unit' in specifications.\n"
            "5. WIRING DIAGRAMS: Every terminal label, wire colour, connection endpoint.\n"
            "6. CROSS-SECTION / CUTAWAY: Name every visible internal component.\n"
            "7. FAULT RELEVANCE: What specific fault or failure mode is shown?\n"
            "8. REPAIR RELEVANCE: What repair task does this image directly support?\n"
            "9. ASSEMBLY ORDER: If numbered steps are visible, list them in order.\n\n"
            "Return ONLY raw JSON — no markdown, no preamble:\n"
            "{\n"
            '  "caption": "<2-4 sentence repair-oriented description — part names,\n'
            '    measurements, assembly order, failure modes, what a technician learns>",\n'
            '  "diagram_type": "<wiring_diagram|exploded_view|sectional_view|\n'
            '    assembly_sequence|flow_chart|torque_table|dimension_drawing|photo|chart|other>",\n'
            '  "components_visible": ["<exact part name as labelled in the manual>"],\n'
            '  "labels_detected": ["<callout number, terminal label, part ref, wire colour>"],\n'
            '  "arrows_point_to": [{"label": "<label text>", "points_to": "<part name>"}],\n'
            '  "relationships": [\n'
            '    {"from": "<part A>", "to": "<part B>", "type": "<relationship_type>"}\n'
            '  ],\n'
            '  "specifications": ["<spec_name: value unit — e.g. capacitor: 25µF, clearance: 0.3mm>"],\n'
            '  "fault_relevance": ["<specific fault or failure mode visible in this image>"],\n'
            '  "repair_relevance": ["<repair task this image directly supports>"],\n'
            '  "assembly_order": ["<step 1>", "<step 2>"],\n'
            '  "search_keywords": ["<term a farmer or technician would search for>"]\n'
            "}"
        )

        # Fix 5: shared thread-safe throttle — all 3 concurrent image workers
        # share this lock, so they never collide on the Gemini rate limit.
        _gemini_throttle()

        response = _get_client().models.generate_content(
            model=_FLASH_MODEL,
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": b64}},
                    {"text": prompt},
                ],
            }],
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
                # Fix 4: Disable Automatic Function Calling — removes ~1-2 s
                # of overhead per request that fires even when no tools are defined.
                "automatic_function_calling": {"disable": True},
            },
        )
        data = json.loads(response.text.strip())
        allowed = {
            "caption", "diagram_type",
            "components_visible", "labels_detected",
            "arrows_point_to", "relationships",
            "specifications", "fault_relevance", "repair_relevance",
            "assembly_order", "search_keywords",
        }
        record.update({k: v for k, v in data.items() if k in allowed})
        # Populate legacy field aliases for backward compatibility with existing code
        # that reads components_shown and connected_relationships
        record["components_shown"]        = record.get("components_visible", [])
        record["connected_relationships"] = [
            {"from": r.get("from",""), "to": r.get("to",""), "relationship": r.get("type","")}
            for r in record.get("relationships", [])
        ]
        record["diagram_type_classifier"] = _classify_diagram_type(
            record.get("diagram_type", "")
        )

    except Exception as e:
        import traceback
        logger.warning("   ⚠️  Captioning failed for %s", record["filename"])
        logger.warning("   Exception type: %s", type(e).__name__)
        logger.warning("   Exception args: %s", e.args)
        logger.warning("   Traceback (last 3 lines):\n%s", 
                    "\n".join(traceback.format_exc().strip().split("\n")[-4:]))
        logger.info("   ✅ %s captioned in %.1fs", record["filename"], _time.time() - _start)
    return record

def caption_images_with_gemini(image_records: list[dict], image_dir: Path) -> list[dict]:
    """
    Caption all image records in parallel using Gemini Vision.

    Fix 6: ThreadPoolExecutor(_CAPTION_WORKERS=3) replaces the sequential loop.
    At 3 workers the throughput is ~3× faster while staying well inside the
    Gemini free-tier rate limit (10 RPM per worker × 3 = 30 RPM max, well
    under the 60 RPM limit for Flash).

    Each worker calls _caption_single_image() which handles its own error
    recovery, so a single failed request never blocks the others.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not image_records:
        return []

    logger.info(
        "   🖼️  Captioning %d images with %d parallel workers...",
        len(image_records), _CAPTION_WORKERS,
    )

    args = [(rec, image_dir) for rec in image_records]

    # executor.map preserves input order — result list matches image_records order.
    with ThreadPoolExecutor(max_workers=_CAPTION_WORKERS) as executor:
        captioned = list(executor.map(_caption_single_image, args))

    logger.info("   ✅ Captioning complete: %d/%d succeeded",
                sum(1 for r in captioned if r.get("caption")), len(captioned))
    return captioned


# ── Per-PDF pipeline ──────────────────────────────────────────────────────────

def process_single_pdf(
    pdf_path:         Path,
    image_output_dir: Path | None = None,
    component_graph:  dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """
    Process one PDF: extract all six knowledge types.

    Returns:
        (procedures, images, specs, fault_entries, diagnostic_trees, repair_procedures)

    All six lists are accumulated by process_directory() and saved to separate JSON files.
    """
    md_text = parse_pdf_to_markdown(str(pdf_path))
    if not md_text:
        logger.warning("⚠️  No text extracted from %s — skipping", pdf_path.name)
        return [], [], [], [], [], []

    chunks = chunk_markdown(md_text)
    logger.info("🗂️  %s → %d chunks to process", pdf_path.name, len(chunks))

    pdf_procedures:       list[dict] = []
    pdf_specs:            list[dict] = []
    pdf_fault_entries:    list[dict] = []
    pdf_diagnostic_trees: list[dict] = []
    pdf_repair_procs:     list[dict] = []
    pdf_tables:           list[dict] = []
    direct_count  = 0
    causal_count  = 0

    for i, chunk_meta in enumerate(chunks):
        is_install  = _is_installation_section(chunk_meta["section"], chunk_meta["text"])
        route_label = "causal" if is_install else "direct"
        logger.info(
            "   🧠 Chunk %d/%d  [%s] → %s",
            i + 1, len(chunks), chunk_meta["section"], route_label,
        )

        extracted = (
            extract_causal_knowledge(chunk_meta["text"])
            if is_install
            else extract_structured_data(chunk_meta["text"])
        )

        # ── Procedures ────────────────────────────────────────────────────────
        if extracted.get("procedures"):
            for proc in extracted["procedures"]:
                proc["manual_source"]  = pdf_path.name
                proc["source_section"] = chunk_meta["section"]
                proc["chunk_id"]       = chunk_meta["chunk_id"]
                proc["knowledge_type"] = (
                    "causal_inferred" if is_install else "direct_repair"
                )
                normalized = _normalize_procedure(proc)

                # Build embedding_text (no raw vector in JSON)
                embed_text = " ".join(filter(None, [
                    normalized.get("machine_family", ""),
                    normalized.get("system", ""),
                    normalized.get("component", ""),
                    " ".join(normalized.get("symptoms", [])),
                    " ".join(normalized.get("causes", [])),
                    " ".join(
                        s.get("instruction", "")
                        for s in normalized.get("step_sequence", [])
                    ),
                    " ".join(normalized.get("if_wrong_installation", [])),
                ]))
                normalized["embedding_text"] = embed_text

                pdf_procedures.append(normalized)
                pdf_specs.extend(_extract_specs_from_procedure(normalized))

            n_procs = len(extracted.get("procedures", []))
            n_faults = len(extracted.get("fault_entries", []))
            n_repairs = len(extracted.get("repair_procedures", []))
            n_tables = len(extracted.get("tables", []))
            n_total = n_procs + n_faults + n_repairs + n_tables
            
            if is_install:
                causal_count += n_total
            else:
                direct_count += n_total
            
            logger.info("      ✅ %d %s items extracted", n_total, route_label)

        # ── Fault entries ─────────────────────────────────────────────────────
        for fe in extracted.get("fault_entries", []):
            fe["manual_source"]  = pdf_path.name
            fe["source_section"] = chunk_meta["section"]
            pdf_fault_entries.append(fe)

        # ── Diagnostic trees ──────────────────────────────────────────────────
        for dt in extracted.get("diagnostic_trees", []):
            dt["manual_source"]  = pdf_path.name
            dt["source_section"] = chunk_meta["section"]
            pdf_diagnostic_trees.append(dt)

        # ── Repair procedures ─────────────────────────────────────────────────
        for rp in extracted.get("repair_procedures", []):
            rp["manual_source"]  = pdf_path.name
            rp["source_section"] = chunk_meta["section"]
            pdf_repair_procs.append(rp)

        # ── ADD THIS: Tables ──────────────────────────────────────────────────
        for tb in extracted.get("tables", []):
            tb["manual_source"]  = pdf_path.name
            pdf_tables.append(tb)

    logger.info(
        "   📊 %s — %d direct_repair + %d causal_inferred = %d procedures "
        "| %d faults | %d trees | %d repair_procs | %d specs | %d tables",
        pdf_path.name, direct_count, causal_count,
        direct_count + causal_count,
        len(pdf_fault_entries), len(pdf_diagnostic_trees),
        len(pdf_repair_procs), len(pdf_specs), len(pdf_tables) # <-- Added tables
    )

    pdf_procedures = deduplicate(pdf_procedures)

    # ── Build page → section map ──────────────────────────────────────────────
    page_section_map: dict[int, str] = {}
    _current_page    = 1
    _current_section = "General"
    for _line in md_text.split("\n"):
        _stripped = _line.strip()
        if re.match(r"^-{4,}$", _stripped):
            _current_page += 1
        if _stripped.startswith("#"):
            _current_section = _stripped.lstrip("#").strip() or _current_section
        if _current_page not in page_section_map:
            page_section_map[_current_page] = _current_section

    # ── Extract and caption images ────────────────────────────────────────────
    img_dir = image_output_dir or (pdf_path.parent / "extracted_images")
    raw_images       = extract_images_from_pdf(pdf_path, img_dir, page_section_map)
    captioned_images = caption_images_with_gemini(raw_images, img_dir) if raw_images else []

    # ── Update component graph (cross-PDF accumulation) ───────────────────────
    if component_graph is not None:
        for proc in pdf_procedures:
            _update_component_graph(component_graph, proc, captioned_images)
        logger.info(
            "   🔗 Component graph: %d nodes after %s",
            len(component_graph), pdf_path.name,
        )

    return (
        pdf_procedures, captioned_images, pdf_specs,
        pdf_fault_entries, pdf_diagnostic_trees, pdf_repair_procs, pdf_tables
    )



# ── Directory-level batch ─────────────────────────────────────────────────────

def process_directory(directory_path: str, output_filename: str) -> None:
    """
    Batch-process all PDFs in a directory.

    Saves 8 JSON databases alongside the PDFs:
      1. <base>.json                      — procedures DB
      2. <base>_images.json               — image knowledge DB
      3. <base>_component_graph.json      — component relationship graph
      4. <base>_spec_database.json        — specification records
      5. <base>_fault_library.json        — fault library (symptom → causes → verify → repair)
      6. <base>_diagnostic_trees.json     — branching yes/no diagnostic trees
      7. <base>_repair_procedures.json    — named standalone repair procedures
      8. <base>_knowledge_index.json      — cross-link index

    Embedding policy:
      All JSON files contain only embedding_text (human-readable string).
      Raw float vectors are NOT saved in JSON — compute them at index time.

    Processing order:
      1. Process each PDF → all six knowledge types
      2. Progressive save after each PDF for procedures + images
      3. Accumulate specs, faults, trees, repair_procs across all PDFs
      4. At end of batch: embed_text + save all auxiliary DBs + knowledge index
    """
    folder = Path(directory_path)
    if not folder.exists() or not folder.is_dir():
        logger.error("❌ Directory not found: %s", directory_path)
        return

    all_pdfs = list(folder.glob("*.pdf"))
    if not all_pdfs:
        logger.warning("⚠️  No PDF files found in '%s'", folder)
        return

    logger.info("🚀 Found %d PDF(s). Starting technician-first extraction...", len(all_pdfs))

    # ── Output paths ──────────────────────────────────────────────────────────
    base            = output_filename.replace(".json", "")
    output_path     = folder / output_filename
    images_path     = folder / f"{base}_images.json"
    graph_path      = folder / f"{base}_component_graph.json"
    spec_path       = folder / f"{base}_spec_database.json"
    tables_path     = folder / f"{base}_tables.json"
    fault_path      = folder / f"{base}_fault_library.json"
    tree_path       = folder / f"{base}_diagnostic_trees.json"
    repair_path     = folder / f"{base}_repair_procedures.json"
    index_path      = folder / f"{base}_knowledge_index.json"
    image_dir       = folder / "extracted_images"

    # ── Accumulators ─────────────────────────────────────────────────────────
    master_procedures:    list[dict]      = []
    master_images:        list[dict]      = []
    master_specs:         list[dict]      = []
    master_faults:        list[dict]      = []
    master_trees:         list[dict]      = []
    master_repair_procs:  list[dict]      = []
    master_tables:        list[dict]      = []
    component_graph:      dict[str, dict] = {}

    # Resume: load existing procedures DB if present
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                master_procedures = json.load(f)
            logger.info("📂 Resuming: %d existing procedures loaded.", len(master_procedures))
        except json.JSONDecodeError:
            logger.warning("⚠️  Existing procedures DB is corrupted — starting fresh.")

    def _pdf_quality_check(pdf_path: Path) -> tuple[bool, str]:
        size_kb = pdf_path.stat().st_size / 1024
        if size_kb < 20:
            return False, f"file too small ({size_kb:.1f} KB — likely corrupt)"
        try:
            import fitz
            doc   = fitz.open(str(pdf_path))
            pages = len(doc)
            doc.close()
            if pages < 3:
                return False, f"only {pages} page(s) — too short for a real manual"
        except Exception as e:
            return False, f"could not open with PyMuPDF: {e}"
        return True, "ok"

    for pdf_file in all_pdfs:
        logger.info("--------------------------------------------------")
        pdf_ok, pdf_reason = _pdf_quality_check(pdf_file)
        if not pdf_ok:
            logger.warning("⏭️  Skipping %s — %s", pdf_file.name, pdf_reason)
            continue
        try:
            new_procs, new_images, new_specs, new_faults, new_trees, new_repairs, new_tables = \
                process_single_pdf(
                    pdf_file,
                    image_output_dir=image_dir,
                    component_graph=component_graph,
                )

            # ── Progressive save: procedures ──────────────────────────────────
            if new_procs:
                master_procedures.extend(new_procs)
                master_procedures = deduplicate(master_procedures)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(master_procedures, f, indent=2, ensure_ascii=False)
                logger.info("💾 Procedures: %d total → %s", len(master_procedures), output_path)

            # ── Progressive save: images ──────────────────────────────────────
            if new_images:
                master_images.extend(new_images)
                with open(images_path, "w", encoding="utf-8") as f:
                    json.dump(master_images, f, indent=2, ensure_ascii=False)
                logger.info("🖼️  Images: %d total → %s", len(master_images), images_path)

            # ── Accumulate remaining outputs ──────────────────────────────────
            master_specs.extend(new_specs)
            master_faults.extend(new_faults)
            master_trees.extend(new_trees)
            master_repair_procs.extend(new_repairs)
            master_tables.extend(new_tables)

        except Exception as e:
            logger.error("❌ Failed to process %s: %s", pdf_file.name, e)

    # ── End-of-batch: build embedding_text and save all auxiliary DBs ─────────
    logger.info(
        "\n🔢 Building embedding_text for %d graph nodes, %d specs, "
        "%d images, %d faults, %d trees, %d repair_procs...",
        len(component_graph), len(master_specs), len(master_images),
        len(master_faults), len(master_trees), len(master_repair_procs),
    )

    if master_tables:
        with open(tables_path, "w", encoding="utf-8") as f:
            json.dump(master_tables, f, indent=2, ensure_ascii=False)
        logger.info("💾 Tables saved: %d → %s", len(master_tables), tables_path)

    _save_component_graph(component_graph, graph_path)
    _save_spec_database(master_specs, spec_path)
    _save_image_knowledge_db(master_images, images_path)
    _save_fault_library(master_faults, fault_path)
    _save_diagnostic_trees(master_trees, tree_path)
    _save_repair_procedures(master_repair_procs, repair_path)
    _build_and_save_knowledge_index(
        master_procedures, master_specs, master_images,
        master_faults, component_graph, index_path,
    )

    logger.info("\n✅ Batch complete!")
    logger.info("   Procedures        → %s  (%d)", output_path,       len(master_procedures))
    logger.info("   Images            → %s  (%d)", images_path,       len(master_images))
    logger.info("   Component graph   → %s  (%d nodes)",  graph_path, len(component_graph))
    logger.info("   Spec database     → %s  (%d specs)",  spec_path,  len(master_specs))
    logger.info("   Fault library     → %s  (%d faults)", fault_path, len(master_faults))
    logger.info("   Diagnostic trees  → %s  (%d trees)",  tree_path,  len(master_trees))
    logger.info("   Repair procedures → %s  (%d)",        repair_path, len(master_repair_procs))
    logger.info("   Knowledge index   → %s",               index_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TARGET_FOLDER = r"D:\AgriFix_Workspace\AgriFixAR_Python_Client\database_creation\Elect_Water_Pump_Pdfs"
    OUTPUT_FILE   = "Master_Water_Pump_DB.json"
    process_directory(TARGET_FOLDER, OUTPUT_FILE)