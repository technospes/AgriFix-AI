"""
add_txt_to_db.py — Incremental TXT Knowledge Adder v1.0
========================================================
Adds LLM-generated .txt solution files into an EXISTING ChromaDB
without touching or rebuilding what's already there.

Usage:
    python add_txt_to_db.py                          # scans ./txt_knowledge/ folder
    python add_txt_to_db.py --txt-dir ./my_txts      # custom folder
    python add_txt_to_db.py --db-dir ./chroma_db     # custom DB path
    python add_txt_to_db.py --dry-run                # parse only, no DB write
    python add_txt_to_db.py --force                  # re-add even if chunk_id exists

Supports the exact chunk format produced by your LLM generation pipeline:

    🔹 Chunk 1 — Some problem title
    PROBLEM: ...
    MACHINE_TYPE: tractor          ← optional, auto-inferred from filename if absent
    TAGS: ...
    SYMPTOM: ...
    LIKELY_CAUSES: ...
    STEPS: ...
    TOOLS: ...
    WARNINGS: ...
    PARTS: ...
    ESCALATE_IF: ...

File → machine_type inference table (when MACHINE_TYPE field is missing):
    *mahindra* / *tractor*       → tractor
    *thresher*                   → thresher
    *harvester*                  → harvester
    *submersible*                → submersible_pump
    *water_pump* / *pump*        → water_pump
    *electric_motor* / *motor*   → electric_motor
    *generator* / *genset*       → generator
    everything else              → universal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("add_txt_to_db.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_CHARS    = 1000
OVERLAP_CHARS = 150
BATCH_SIZE   = 64

# Filename → machine_type inference (order matters: more specific first)
_FILENAME_MACHINE_MAP: List[Tuple[str, str]] = [
    ("submersible",    "submersible_pump"),
    ("water_pump",     "water_pump"),
    ("electric_motor", "electric_motor"),
    ("electric",       "electric_motor"),
    ("thresher",       "thresher"),
    ("harvester",      "harvester"),
    ("mahindra",       "tractor"),
    ("tractor",        "tractor"),
    ("generator",      "generator"),
    ("genset",         "generator"),
    ("pump",           "water_pump"),
    ("motor",          "electric_motor"),
]

# MACHINE_TYPE: field value aliases → canonical IDs
_MACHINE_ALIASES: Dict[str, str] = {
    "tractor":          "tractor",
    "mahindra":         "tractor",
    "thresher":         "thresher",
    "multi crop":       "thresher",
    "multicrop":        "thresher",
    "harvester":        "harvester",
    "combine":          "harvester",
    "water pump":       "water_pump",
    "water_pump":       "water_pump",
    "submersible":      "submersible_pump",
    "submersible_pump": "submersible_pump",
    "borewell":         "submersible_pump",
    "electric motor":   "electric_motor",
    "electric_motor":   "electric_motor",
    "motor":            "electric_motor",
    "generator":        "generator",
    "genset":           "generator",
    "diesel engine":    "diesel_engine",
    "power tiller":     "power_tiller",
}

FAILURE_TAXONOMY: Dict[str, List[str]] = {
    "electrical":  ["wiring","voltage","short","relay","fuse","mcb","capacitor",
                    "battery","alternator","motor winding","bijli","current"],
    "mechanical":  ["bearing","shaft","gear","coupling","belt","pulley","impeller",
                    "piston","crankshaft","camshaft","valve","spring"],
    "hydraulic":   ["hydraulic","cylinder","control valve","3-point","lift","hitch",
                    "oil pressure","pump pressure"],
    "thermal":     ["overheat","temperature","radiator","coolant","thermostat","garam"],
    "lubrication": ["oil level","grease","lubricant","dry bearing","oil change","viscosity"],
    "cavitation":  ["cavitation","air lock","suction","prime","hawa","vacuum"],
    "corrosion":   ["corrosion","rust","oxidation","white powder","terminal"],
    "seal_failure":["mechanical seal","shaft seal","lip seal","packing","gland"],
    "blockage":    ["blockage","clog","choked","filter","strainer","jammed"],
    "fuel":        ["fuel","diesel","petrol","injector","carburetor","injection pump"],
    "ignition":    ["spark","ignition","glow plug","decompression"],
    "alignment":   ["misalignment","vibration","wobble","runout","balance"],
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _infer_machine_from_filename(filename: str) -> str:
    fn = filename.lower()
    for keyword, machine_id in _FILENAME_MACHINE_MAP:
        if keyword in fn:
            return machine_id
    return "universal"


def _resolve_machine_type(raw_field: str, filename: str) -> str:
    """Resolve MACHINE_TYPE: field value to canonical ID, falling back to filename."""
    if raw_field:
        val = raw_field.strip().lower().replace("-", " ")
        # Direct alias lookup
        if val in _MACHINE_ALIASES:
            return _MACHINE_ALIASES[val]
        # Partial match
        for alias, canonical in _MACHINE_ALIASES.items():
            if alias in val or val in alias:
                return canonical
    return _infer_machine_from_filename(filename)


def _infer_failure_taxonomy(text: str) -> List[str]:
    tl = text.lower()
    result = [cat for cat, kws in FAILURE_TAXONOMY.items() if any(kw in tl for kw in kws)]
    return result or ["mechanical"]


def _infer_risk_level(text: str) -> str:
    tl = text.lower()
    electrical = any(kw in tl for kw in ["electric","voltage","current","mcb","capacitor","winding","bijli"])
    rotating   = any(kw in tl for kw in ["belt","pulley","shaft","impeller","pto","rotating","fan","flywheel"])
    fuel_fire  = any(kw in tl for kw in ["fuel","diesel","petrol","fire","spark","flammable"])
    water      = any(kw in tl for kw in ["water","coolant","flood","submersible","wet"])
    if electrical and rotating:
        return "CRITICAL"
    if electrical or fuel_fire:
        return "HIGH"
    if rotating or water:
        return "MEDIUM"
    return "LOW"


def _parse_list_field(raw: str) -> List[str]:
    if not raw:
        return []
    return sorted(set(item.strip().lower() for item in re.split(r"[,;]+", raw) if item.strip()))


def _split_with_overlap(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        for sep in ("\n\n", "\n", ". ", "! ", "? "):
            pos = text.rfind(sep, start, end)
            if pos > start:
                end = pos + 1
                break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end - overlap)
    seen, result = set(), []
    for c in chunks:
        key = re.sub(r"\s+", " ", c).strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(c)
    return result


# ── Chunk parser ───────────────────────────────────────────────────────────────

@dataclass
class TxtChunk:
    chunk_id:    str
    problem:     str
    machine_type: str
    tags:        List[str]
    parts:       List[str]
    escalate_if: str
    content:     str
    source_file: str
    failure_taxonomy: List[str] = field(default_factory=list)
    risk_level:  str = "LOW"
    electrical_hazard: bool = False
    shutdown_required: bool = False

    def content_hash(self) -> str:
        return hashlib.md5(re.sub(r"\s+", " ", self.content).strip().lower().encode()).hexdigest()

    def to_document_metadata(self) -> dict:
        """Produce a ChromaDB-compatible metadata dict (all values scalar or str)."""
        return {
            "chunk_id":          self.chunk_id,
            "problem":           self.problem[:200],
            "machine_type":      self.machine_type,
            "tags":              json.dumps(self.tags),
            "parts":             json.dumps(self.parts),
            "escalate_if":       self.escalate_if[:300],
            "source_file":       self.source_file,
            "failure_taxonomy":  json.dumps(self.failure_taxonomy),
            "risk_level":        self.risk_level,
            "electrical_hazard": self.electrical_hazard,
            "shutdown_required": self.shutdown_required,
            "content_hash":      self.content_hash(),
            "source":            "llm_generated_txt",
        }


# Field patterns matching the exact format used in all 4 txt files
_CHUNK_BOUNDARY = re.compile(
    r"🔹\s*Chunk\s+[\w\d]+\s*[—–\-]+\s*(.+?)(?=\n|$)", re.IGNORECASE
)

_FIELDS = {
    "problem":      re.compile(r"^PROBLEM\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "machine_type": re.compile(r"^MACHINE_TYPE\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "tags":         re.compile(r"^TAGS\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "symptom":      re.compile(r"^SYMPTOM\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "likely_causes":re.compile(r"^LIKELY_CAUSES\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "mech_reason":  re.compile(r"^MECHANICAL_REASON\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "diag_branch":  re.compile(r"^DIAGNOSTIC_BRANCH\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "steps":        re.compile(r"^STEPS\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "tools":        re.compile(r"^TOOLS\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "warnings":     re.compile(r"^WARNINGS?\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "visual":       re.compile(r"^VISUAL\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "parts":        re.compile(r"^PARTS\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    "escalate_if":  re.compile(r"^ESCALATE_IF\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
}


def _get(text: str, field: str) -> str:
    m = _FIELDS[field].search(text)
    return m.group(1).strip() if m else ""


def parse_txt_file(file_path: Path) -> Tuple[List[TxtChunk], List[str]]:
    """
    Parse a .txt knowledge file into TxtChunk objects.
    Returns (chunks, rejection_reasons).
    """
    text = file_path.read_text(encoding="utf-8")
    filename = file_path.name
    chunks: List[TxtChunk] = []
    rejections: List[str] = []

    # Split on chunk boundaries
    boundaries = list(_CHUNK_BOUNDARY.finditer(text))
    if not boundaries:
        rejections.append(f"{filename}: no '🔹 Chunk' boundaries found")
        return chunks, rejections

    for i, match in enumerate(boundaries):
        title    = match.group(1).strip()
        start    = match.end()
        end      = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
        body     = text[start:end].strip()

        problem  = _get(body, "problem")
        if not problem:
            rejections.append(f"{filename} [{title}]: missing PROBLEM field — skipped")
            continue

        raw_mt   = _get(body, "machine_type")
        machine  = _resolve_machine_type(raw_mt, filename)

        tags      = _parse_list_field(_get(body, "tags"))
        parts     = _parse_list_field(_get(body, "parts"))
        esc       = _get(body, "escalate_if")
        symptom   = _get(body, "symptom")
        causes    = _get(body, "likely_causes")
        mech      = _get(body, "mech_reason")
        diag      = _get(body, "diag_branch")
        steps     = _get(body, "steps")
        tools     = _get(body, "tools")
        warnings  = _get(body, "warnings")

        # Build the page_content that goes into ChromaDB
        # Mirrors the format build_knowledge.py uses so the LLM sees the same layout
        content_parts = [
            f"PROBLEM: {problem}",
            f"MACHINE: {machine}",
        ]
        if symptom:
            content_parts.append(f"SYMPTOM: {symptom}")
        if causes:
            content_parts.append(f"LIKELY_CAUSES: {causes}")
        if mech:
            content_parts.append(f"MECHANICAL_REASON: {mech}")
        if diag:
            content_parts.append(f"DIAGNOSTIC_BRANCH: {diag}")
        if steps:
            content_parts.append(f"STEPS: {steps}")
        if tools:
            content_parts.append(f"TOOLS: {tools}")
        if warnings:
            content_parts.append(f"⚠️ WARNINGS: {warnings}")
        if esc:
            content_parts.append(f"ESCALATE_IF: {esc}")
        if parts:
            content_parts.append(f"PARTS: {', '.join(parts)}")
        content_parts.append(f"SOURCE: {filename}")

        content_raw = "\n".join(content_parts)
        full_text   = (problem + " " + " ".join(tags) + " " + causes + " " + warnings).lower()

        taxonomy  = _infer_failure_taxonomy(full_text)
        risk      = _infer_risk_level(full_text)
        elec_haz  = any(kw in full_text for kw in ["electric","voltage","bijli","mcb","capacitor","winding"])
        shutdown  = any(kw in full_text for kw in ["switch off","turn off","disconnect power","stop engine","isolate"])

        # Deterministic chunk_id: hash of (machine_type + problem + filename)
        content_hash_for_id = hashlib.md5(content_raw.encode()).hexdigest()[:8]
        raw_id   = f"{machine}|{problem}|{filename}|{title}|{content_hash_for_id}"
        chunk_id = f"txt_{hashlib.md5(raw_id.encode()).hexdigest()[:14]}"

        # If content is long, split with overlap (creates _0, _1 … sub-chunks)
        sub_texts = _split_with_overlap(content_raw)
        for j, sub in enumerate(sub_texts):
            cid = chunk_id if j == 0 else f"{chunk_id}_{j}"
            chunks.append(TxtChunk(
                chunk_id        = cid,
                problem         = problem,
                machine_type    = machine,
                tags            = tags,
                parts           = parts,
                escalate_if     = esc,
                content         = sub,
                source_file     = filename,
                failure_taxonomy= taxonomy,
                risk_level      = risk,
                electrical_hazard = elec_haz,
                shutdown_required = shutdown,
            ))

    logger.info("  Parsed %s → %d chunks, %d rejected", filename, len(chunks), len(rejections))
    return chunks, rejections


# ── Incremental DB writer ──────────────────────────────────────────────────────

def _get_existing_ids(db) -> Set[str]:
    """Pull all chunk_ids already in ChromaDB."""
    try:
        result = db._collection.get(include=["metadatas"])
        ids: Set[str] = set()
        for meta in result.get("metadatas") or []:
            cid = (meta or {}).get("chunk_id", "")
            if cid:
                ids.add(cid)
        return ids
    except Exception as exc:
        logger.warning("Could not read existing IDs: %s — will rely on content hash dedup", exc)
        return set()


def _get_existing_content_hashes(db) -> Set[str]:
    """Pull all content_hash values already in the DB (secondary dedup guard)."""
    try:
        result = db._collection.get(include=["metadatas"])
        return {
            (meta or {}).get("content_hash", "")
            for meta in result.get("metadatas") or []
            if (meta or {}).get("content_hash")
        }
    except Exception:
        return set()


def add_txt_to_chroma(
    txt_dir: Path,
    db_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Main entry point. Parses all .txt files in txt_dir and adds new chunks
    to the existing ChromaDB at db_dir.

    Returns a stats dict.
    """
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings

    stats = {
        "files_scanned":  0,
        "chunks_parsed":  0,
        "chunks_skipped": 0,   # already in DB
        "chunks_added":   0,
        "rejections":     [],
    }

    # ── Collect .txt files ────────────────────────────────────────────────────
    txt_files = sorted(txt_dir.rglob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in %s", txt_dir)
        return stats
    logger.info("Found %d .txt file(s) in %s", len(txt_files), txt_dir)

    # ── Parse all files ───────────────────────────────────────────────────────
    all_chunks: List[TxtChunk] = []
    for fp in txt_files:
        stats["files_scanned"] += 1
        parsed, rejected = parse_txt_file(fp)
        all_chunks.extend(parsed)
        stats["rejections"].extend(rejected)

    stats["chunks_parsed"] = len(all_chunks)
    if not all_chunks:
        logger.warning("No chunks parsed. Check file format (needs 🔹 Chunk boundaries).")
        return stats

    if dry_run:
        logger.info("DRY RUN — %d chunks would be added (no DB write)", len(all_chunks))
        _print_summary(all_chunks, stats)
        return stats

    # ── Connect to existing ChromaDB ─────────────────────────────────────────
    if not db_dir.exists():
        logger.error("ChromaDB directory not found: %s", db_dir)
        logger.error("Run build_knowledge.py first to create the DB, then run this script.")
        sys.exit(1)

    logger.info("Loading embedding model (BAAI/bge-m3) …")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(persist_directory=str(db_dir), embedding_function=embeddings)
    logger.info("Connected to ChromaDB at %s", db_dir)

    # ── Deduplication ─────────────────────────────────────────────────────────
    if not force:
        existing_ids     = _get_existing_ids(db)
        existing_hashes  = _get_existing_content_hashes(db)
        logger.info("DB already contains %d chunk IDs", len(existing_ids))
    else:
        existing_ids, existing_hashes = set(), set()
        logger.info("--force: skipping dedup check, re-adding all chunks")

    new_chunks: List[TxtChunk] = []
    for chunk in all_chunks:
        if chunk.chunk_id in existing_ids:
            stats["chunks_skipped"] += 1
            logger.debug("Skip (id exists): %s", chunk.chunk_id)
        elif chunk.content_hash() in existing_hashes:
            stats["chunks_skipped"] += 1
            logger.debug("Skip (content dup): %s", chunk.chunk_id)
        else:
            new_chunks.append(chunk)

    if not new_chunks:
        logger.info("Nothing new to add — all %d chunks already in DB.", len(all_chunks))
        return stats

    logger.info("%d new chunks to add (%d already existed)", len(new_chunks), stats["chunks_skipped"])

    # ── Convert to Documents ──────────────────────────────────────────────────
    documents = [
        Document(page_content=c.content, metadata=c.to_document_metadata())
        for c in new_chunks
    ]

    # ── Embed and insert in batches ───────────────────────────────────────────
    import time
    total   = len(documents)
    n_batch = (total + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("Inserting %d documents in %d batch(es) of %d …", total, n_batch, BATCH_SIZE)

    for b, i in enumerate(range(0, total, BATCH_SIZE), start=1):
        batch = documents[i : i + BATCH_SIZE]
        try:
            db.add_documents(batch)
            logger.info("  Batch %d/%d — inserted %d chunks", b, n_batch, len(batch))
            stats["chunks_added"] += len(batch)
        except Exception as exc:
            logger.error("  Batch %d failed: %s — retrying once …", b, exc)
            time.sleep(5)
            db.add_documents(batch)
            stats["chunks_added"] += len(batch)
        if b < n_batch:
            time.sleep(1)

    _print_summary(all_chunks, stats)
    return stats


def _print_summary(chunks: List[TxtChunk], stats: dict) -> None:
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("  Files scanned : %d", stats["files_scanned"])
    logger.info("  Chunks parsed : %d", stats["chunks_parsed"])
    logger.info("  Chunks skipped: %d (already in DB)", stats["chunks_skipped"])
    logger.info("  Chunks added  : %d", stats["chunks_added"])
    logger.info("  Rejections    : %d", len(stats["rejections"]))

    # Per-machine breakdown
    from collections import Counter
    machines = Counter(c.machine_type for c in chunks)
    logger.info("  Machine types : %s", dict(machines))

    if stats["rejections"]:
        logger.warning("  Rejected chunks:")
        for r in stats["rejections"]:
            logger.warning("    %s", r)
    logger.info("=" * 70)


# ── LLM prompt helpers (bonus — for generating more txt files) ────────────────

GENERATION_PROMPT_TEMPLATE = """
You are generating diagnostic knowledge chunks for an agricultural machinery RAG system.

Generate {n_chunks} fault-diagnosis chunks for: {machine_type}
Problem area: {problem_area}

Each chunk MUST follow this EXACT format (copy the field names exactly):

🔹 Chunk {number} — {short_title}
PROBLEM: {one_line_problem_statement}
MACHINE_TYPE: {machine_type_id}
TAGS: {comma_separated_tags_including_hindi_transliterations}
SYMPTOM: {what_the_farmer_observes}
LIKELY_CAUSES: {comma_separated_causes}
MECHANICAL_REASON: {one_sentence_technical_explanation}
DIAGNOSTIC_BRANCH: IF {observable_condition} -> {route}; IF {other_condition} -> {other_route}
STEPS: {step1}; {step2}; {step3}
TOOLS: {tool1}, {tool2}
WARNINGS: {safety_warning}
VISUAL: {ComponentName1}, {ComponentName2}
PARTS: {PartName1}, {PartName2}
ESCALATE_IF: {condition_requiring_technician}

Rules:
- TAGS must include Hinglish phrases farmers actually say (e.g. "chalu nahi", "garam ho raha")
- STEPS must be actionable without special tools unless TOOLS field lists them
- ESCALATE_IF must be a real escalation condition, not a generic phrase
- MACHINE_TYPE must be one of: tractor, thresher, harvester, water_pump, submersible_pump,
  electric_motor, generator, power_tiller, diesel_engine
- Keep each chunk under 800 characters total
""".strip()


def print_generation_prompt(machine_type: str, problem_area: str, n_chunks: int = 10) -> None:
    """Print a ready-to-use LLM prompt for generating more chunks."""
    print(GENERATION_PROMPT_TEMPLATE.format(
        machine_type=machine_type,
        machine_type_id=machine_type.lower().replace(" ", "_"),
        problem_area=problem_area,
        n_chunks=n_chunks,
        number="N",
        short_title="<short description>",
        one_line_problem_statement="<problem>",
        comma_separated_tags_including_hindi_transliterations="<tags>",
        what_the_farmer_observes="<symptom>",
        comma_separated_causes="<causes>",
        one_sentence_technical_explanation="<reason>",
        observable_condition="<condition>",
        route="<route>",
        other_condition="<other>",
        other_route="<other_route>",
        step1="<step1>",
        step2="<step2>",
        step3="<step3>",
        tool1="<tool>",
        tool2="<tool>",
        safety_warning="<warning>",
        ComponentName1="<component>",
        ComponentName2="<component>",
        PartName1="<part>",
        PartName2="<part>",
        condition_requiring_technician="<condition>",
    ))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Incrementally add LLM-generated .txt files to an existing ChromaDB"
    )
    parser.add_argument(
        "--txt-dir", default="./txt_knowledge",
        help="Folder containing .txt knowledge files (default: ./txt_knowledge)"
    )
    parser.add_argument(
        "--db-dir", default="./chroma_db",
        help="Existing ChromaDB directory (default: ./chroma_db)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and report only — do not write to DB"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-add chunks even if chunk_id already exists in DB"
    )
    parser.add_argument(
        "--prompt", nargs=2, metavar=("MACHINE_TYPE", "PROBLEM_AREA"),
        help="Print an LLM generation prompt and exit. E.g.: --prompt tractor overheating"
    )
    args = parser.parse_args()

    if args.prompt:
        print_generation_prompt(args.prompt[0], args.prompt[1])
        return

    txt_dir = Path(args.txt_dir)
    db_dir  = Path(args.db_dir)

    if not txt_dir.exists():
        logger.error("txt-dir not found: %s", txt_dir)
        logger.error("Create the folder and put your .txt files inside it.")
        sys.exit(1)

    stats = add_txt_to_chroma(txt_dir, db_dir, dry_run=args.dry_run, force=args.force)

    if stats["chunks_added"] > 0:
        logger.info("✅ Done. %d new chunks added to %s", stats["chunks_added"], db_dir)
    elif not args.dry_run:
        logger.info("ℹ️  No new chunks added (all already present). Use --force to override.")

    sys.exit(0 if not stats["rejections"] else 1)


if __name__ == "__main__":
    main()