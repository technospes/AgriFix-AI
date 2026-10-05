import os
import re
import json
import hashlib
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# FIX 2: Dense retrieval narrative — replaces thin _generate_retrieval_narrative
from dense_narrative import dense_retrieval_narrative as _generate_retrieval_narrative

# FIX 7: Clean metadata schema — enforces ChromaDB field size limits
from metadata_schema import clean_to_metadata

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("knowledge_build.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# CAP CHUNK SIZE: Ensures embeddings remain dense and don't hit model limits
MAX_CHARS = 1000
OVERLAP_CHARS = 150


def split_with_overlap(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> List[str]:
    """
    Split text into overlapping chunks with sliding window.
    Prefers paragraph > newline > sentence boundaries before hard split.
    Returns single-element list if text fits in one chunk.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Prefer paragraph break
        split_pos = text.rfind("\n\n", start, end)
        if split_pos == -1 or split_pos <= start:
            # Prefer single newline
            split_pos = text.rfind("\n", start, end)
        if split_pos == -1 or split_pos <= start:
            # Prefer sentence end
            for punct in (". ", "! ", "? "):
                p = text.rfind(punct, start, end)
                if p > start:
                    split_pos = p + 1
                    break
        if split_pos <= start:
            # Hard split
            split_pos = end

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        # Slide back by overlap
        start = max(start + 1, split_pos - overlap)

    # Remove empty / duplicate chunks
    seen: set = set()
    result: List[str] = []
    for c in chunks:
        key = re.sub(r"\s+", " ", c).strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(c)
    return result

# Universal problem categories for auto-tagging
CATEGORY_KEYWORDS = {
    "not_starting": ["not start", "won't start", "no start", "dead", "crank", "starter", "self", "battery"],
    "noise": ["noise", "sound", "knocking", "clicking", "grinding", "rattling", "squealing", "awaaz"],
    "leaking": ["leak", "dripping", "seeping", "oil leak", "tapak"],
    "overheating": ["overheat", "hot", "boiling", "temperature", "garam", "radiator"],
    "vibration": ["vibrat", "shaking", "wobbl", "shake", "hil"],
    "smoke": ["smoke", "fumes", "dhuan"],
    "power_loss": ["low power", "no power", "weak", "sluggish", "power loss", "pickup"],
    "electrical": ["battery", "wiring", "fuse", "relay", "alternator", "voltage", "bijli"],
    "fuel": ["fuel", "diesel", "petrol", "injector", "carburetor", "pump"],
    "water_flow": ["water", "flow", "pressure", "discharge", "pani"],
    "hydraulic": ["hydraulic", "lift", "3 point", "cylinder"],
    "transmission": ["gear", "clutch", "pto", "transmission"],
}


# ═══════════════════════════════════════════════════════════════════════════
# OBJECTIVE 6 — FAILURE TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════

FAILURE_TAXONOMY: Dict[str, List[str]] = {
    "electrical":     ["wiring", "voltage", "short", "relay", "fuse", "mcb", "capacitor",
                       "battery", "alternator", "motor winding", "bijli", "current"],
    "mechanical":     ["bearing", "shaft", "gear", "coupling", "belt", "pulley", "impeller",
                       "piston", "crankshaft", "camshaft", "valve", "spring"],
    "hydraulic":      ["hydraulic", "cylinder", "control valve", "3-point", "lift", "hitch",
                       "oil pressure", "pump pressure"],
    "pneumatic":      ["pneumatic", "air pressure", "compressor", "air line", "bleed"],
    "thermal":        ["overheat", "temperature", "radiator", "coolant", "thermostat",
                       "cooling", "garam", "heat"],
    "lubrication":    ["oil level", "grease", "lubricant", "dry bearing", "oil change",
                       "viscosity", "oil grade"],
    "wear":           ["worn", "wear", "erosion", "clearance exceeded", "play", "backlash"],
    "fluid":          ["leaking", "seeping", "dripping", "seal leak", "gasket", "o-ring",
                       "tapak", "paani"],
    "power_supply":   ["no power", "voltage drop", "phase failure", "single phase", "supply"],
    "alignment":      ["misalignment", "vibration", "wobble", "runout", "hil", "balance"],
    "cavitation":     ["cavitation", "air lock", "suction", "prime", "hawa", "vacuum"],
    "corrosion":      ["corrosion", "rust", "oxidation", "white powder", "terminal"],
    "bearing_failure":["bearing noise", "bearing seizure", "race damage", "ball bearing"],
    "seal_failure":   ["mechanical seal", "shaft seal", "lip seal", "packing", "gland"],
    "blockage":       ["blockage", "clog", "choked", "filter", "strainer", "jammed", "band"],
}

def _infer_failure_taxonomy(text: str) -> List[str]:
    text_lower = text.lower()
    return [
        category for category, keywords in FAILURE_TAXONOMY.items()
        if any(kw in text_lower for kw in keywords)
    ] or ["mechanical"]

_MACHINE_TYPE_FIELD_PRIORITY = [
    "machine_type",
    "machine_family",
    "manual_source",
]

def _resolve_machine_type(record: dict) -> str:
    mt = str(record.get("machine_type") or "").strip().lower()
    if mt and mt not in ("", "universal", "none"):
        return mt

    mf = str(record.get("machine_family") or "").strip().lower()
    if mf and mf not in ("", "none"):
        return re.sub(r"\s+", "_", mf)

    src = str(record.get("manual_source") or "").strip().lower()
    if src:
        stem = re.sub(r"\.[a-z]{2,5}$", "", src)
        words = re.findall(r"[a-z]+", stem)
        words = [w for w in words if len(w) > 2 and not w.isdigit()]
        if words:
            return "_".join(words[:2])

    return "universal"


# ═══════════════════════════════════════════════════════════════════════════
# OBJECTIVE 7 — SAFETY METADATA SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

def _build_safety_metadata(proc: dict, content: str) -> dict:
    text  = content.lower()

    electrical_hazard = any(kw in text for kw in [
        "electric", "wiring", "voltage", "current", "short circuit",
        "relay", "mcb", "fuse", "battery terminal", "capacitor",
    ])
    rotating_parts = any(kw in text for kw in [
        "belt", "pulley", "shaft", "impeller", "blade", "pto", "rotating",
        "fan", "flywheel", "gear",
    ])
    fuel_fire = any(kw in text for kw in [
        "fuel", "diesel", "petrol", "flammable", "fire", "ignition", "spark",
    ])
    water_exposure = any(kw in text for kw in [
        "water", "coolant", "flooding", "submersible", "wet", "fluid leak",
    ])
    shutdown_required = any(kw in text for kw in [
        "turn off", "switch off", "disconnect power", "stop engine",
        "isolate", "depressurise", "drain",
    ])
    expert_required = any(kw in text for kw in [
        "certified mechanic", "service centre", "professional", "technician",
        "do not attempt", "escalate",
    ])

    if electrical_hazard and rotating_parts:
        risk_level = "CRITICAL"
    elif electrical_hazard or fuel_fire:
        risk_level = "HIGH"
    elif rotating_parts or water_exposure:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_level":             risk_level,
        "shutdown_required":      shutdown_required,
        "electrical_hazard":      electrical_hazard,
        "rotating_parts_warning": rotating_parts,
        "water_exposure_risk":    water_exposure,
        "fuel_fire_risk":         fuel_fire,
        "expert_required":        expert_required,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURED CHUNK PARSER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedChunk:
    chunk_id: str
    problem: str
    machine_type: str
    tags: List[str]
    problem_categories: List[str]
    parts: List[str]
    escalate_if: str
    content: str
    source_file: str
    has_repair_refs: bool = False
    failure_taxonomy: List[str] = None          # type: ignore[assignment]
    safety_metadata:  Dict      = None            # type: ignore[assignment]

    def __post_init__(self):
        if self.failure_taxonomy is None:
            self.failure_taxonomy = _infer_failure_taxonomy(
                self.problem + " " + " ".join(self.tags)
            )
        if self.safety_metadata is None:
            self.safety_metadata = _build_safety_metadata({}, self.content)

    def to_metadata(self) -> Dict:
        meta = clean_to_metadata(self)
        meta["has_repair_refs"] = bool("RELATED REPAIR PROCEDURES:" in self.content)
        return meta

class StructuredChunkParser:
    CHUNK_BOUNDARY = re.compile(r"🔹\s*Chunk\s+\[?([\w\d_]+)\]?\s*[—–-]\s*(.+?)(?=\n|$)", re.IGNORECASE)
    
    FIELD_PATTERNS = {
        "problem": re.compile(r"(?:PROBLEM|Problem)\s*:\s*(.+?)(?=\n(?:MACHINE_TYPE|TAGS|SYMPTOM|LIKELY_CAUSES|PROBLEM_CATEGORIES|PARTS|ESCALATE_IF|CAUSE|FIX|$))", re.IGNORECASE | re.DOTALL),
        "machine_type": re.compile(r"(?:MACHINE_TYPE|Machine_Type)\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE),
        "tags": re.compile(r"(?:TAGS|Tags)\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE),
        "problem_categories": re.compile(r"(?:PROBLEM_CATEGORIES|Problem_Categories)\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE),
        "parts": re.compile(r"(?:PARTS|Parts)\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE),
        "escalate_if": re.compile(r"(?:ESCALATE_IF|Escalate_If)\s*:\s*(.+?)(?=\n(?:🔹|$))", re.IGNORECASE | re.DOTALL),
    }
    
    def __init__(self):
        self.rejected_chunks: List[Tuple[str, str]] = []
        
    def parse_file(self, file_path: Path) -> List[ParsedChunk]:
        logger.info(f"📖 Parsing: {file_path.name}")
        text = file_path.read_text(encoding="utf-8")
        
        chunks_raw = self._split_chunks(text)
        parsed = []
        for chunk_id, chunk_text in chunks_raw:
            try:
                chunk_obj = self._parse_chunk(chunk_id, chunk_text, file_path.name)
                if chunk_obj:
                    parsed.append(chunk_obj)
            except Exception as e:
                self.rejected_chunks.append((chunk_id, f"Parse error: {e}"))
        
        logger.info(f"   ✓ {file_path.name} → {len(parsed)} valid chunks, {len(chunks_raw) - len(parsed)} rejected")
        return parsed
    
    def _split_chunks(self, text: str) -> List[Tuple[str, str]]:
        chunks = []
        matches = list(self.CHUNK_BOUNDARY.finditer(text))
        for i, match in enumerate(matches):
            chunk_id = match.group(1).strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunks.append((chunk_id, text[start_pos:end_pos].strip()))
        return chunks
    
    def _parse_chunk(self, chunk_id: str, text: str, source_file: str) -> Optional[ParsedChunk]:
        problem_match = self.FIELD_PATTERNS["problem"].search(text)
        machine_match = self.FIELD_PATTERNS["machine_type"].search(text)
        
        if not problem_match:
            self.rejected_chunks.append((chunk_id, "Missing PROBLEM field"))
            return None
            
        problem = problem_match.group(1).strip()
        
        if machine_match:
            machine_type = machine_match.group(1).strip().lower()
        else:
            machine_type = _resolve_machine_type({"manual_source": source_file})
            
        tags_match = self.FIELD_PATTERNS["tags"].search(text)
        tags = self._parse_list_field(tags_match.group(1) if tags_match else "")
        
        cats_match = self.FIELD_PATTERNS["problem_categories"].search(text)
        problem_categories = self._parse_list_field(cats_match.group(1) if cats_match else "")
        
        if not problem_categories:
            combined_text = f"{problem} {' '.join(tags)}".lower()
            inferred = []
            for cat, kws in CATEGORY_KEYWORDS.items():
                if any(kw in combined_text for kw in kws):
                    inferred.append(cat)
            problem_categories = inferred
            
        parts_match = self.FIELD_PATTERNS["parts"].search(text)
        parts = self._parse_list_field(parts_match.group(1) if parts_match else "")
        
        escalate_match = self.FIELD_PATTERNS["escalate_if"].search(text)
        escalate_if = escalate_match.group(1).strip() if escalate_match else ""
        
        content_clean = text
        for pattern in self.FIELD_PATTERNS.values():
            content_clean = pattern.sub("", content_clean)
        content_clean = f"PROBLEM: {problem}\nTAGS: {' '.join(tags)}\nESCALATE_IF: {escalate_if}\n\n{content_clean}"
        content_clean = re.sub(r"\n{3,}", "\n\n", content_clean).strip()
        content_clean = content_clean[:MAX_CHARS] # Cap size
        
        return ParsedChunk(
            chunk_id=chunk_id, problem=problem, machine_type=machine_type,
            tags=tags, problem_categories=problem_categories, parts=parts,
            escalate_if=escalate_if, content=content_clean, source_file=source_file
        )
    
    @staticmethod
    def _parse_list_field(raw: str) -> List[str]:
        if not raw: return []
        return sorted(set([item.strip().lower() for item in re.split(r"[,;]+", raw) if item.strip()]))

# ═══════════════════════════════════════════════════════════════════════════
# JSON DATABASE BRIDGES
# ═══════════════════════════════════════════════════════════════════════════

def _infer_problem_categories(proc: dict) -> List[str]:
    text = (
        proc.get("system", "") + " " +
        " ".join(proc.get("symptoms", [])) + " " +
        " ".join(proc.get("causes", []))
    ).lower()
    cats = [cat for cat, kws in CATEGORY_KEYWORDS.items() if any(kw in text for kw in kws)]
    return cats or ["general"]


def _synthetic_variants(symptoms: List[str], causes: List[str], component: str) -> str:
    """
    Generate lightweight searchable query variants from existing content only.
    Appended to chunk text to improve noisy/paraphrased query retrieval.
    No hallucination — only rephrases available strings.
    """
    lines: List[str] = []
    for sym in symptoms[:4]:
        sl = sym.lower().strip()
        if sl:
            lines.append(sl)
            # Simple alias variants
            if "not start" in sl or "won't start" in sl:
                lines.append("machine not starting dead no crank")
            if "hum" in sl:
                lines.append("motor humming buzzing no rotation")
            if "overheat" in sl or "hot" in sl:
                lines.append("overheating excessive heat temperature high")
            if "vibrat" in sl or "shak" in sl:
                lines.append("vibration shaking wobbling unbalanced")
            if "no water" in sl or "no flow" in sl or "no discharge" in sl:
                lines.append("no water flow low discharge paani nahi")
            if "noise" in sl or "sound" in sl:
                lines.append("abnormal noise knocking grinding sound")
            if "smoke" in sl:
                lines.append("smoke fumes dhuan burning smell")
    for cause in causes[:3]:
        cl = cause.lower().strip()
        if cl:
            lines.append(cl)
    if component:
        lines.append(f"{component.lower()} fault repair replace")
    return "SEARCH_VARIANTS: " + " | ".join(dict.fromkeys(lines)) if lines else ""


def _make_proc_chunk(
    chunk_id: str,
    subtype: str,
    content_raw: str,
    proc: dict,
    source_file: str,
    component: str,
    machine_type: str,
    system: str,
    symptoms_str: str,
    tags: List[str],
    problem_categories: List[str],
    repair_refs: List[str],
    escalate_if: str,
) -> ParsedChunk:
    content = re.sub(r"\n{3,}", "\n\n", content_raw).strip()
    sub_chunks = split_with_overlap(content)
    # Use first sub_chunk for this record; caller iterates if needed
    content_final = sub_chunks[0] if sub_chunks else content[:MAX_CHARS]
    return ParsedChunk(
        chunk_id          = f"{chunk_id}_{subtype}",
        problem           = f"{component} — {symptoms_str[:100]}",
        machine_type      = machine_type,
        tags              = tags,
        problem_categories= problem_categories,
        parts             = [component] + proc.get("part_numbers", []),
        escalate_if       = escalate_if,
        content           = content_final,
        source_file       = source_file,
        has_repair_refs   = bool(repair_refs),
        failure_taxonomy  = _infer_failure_taxonomy(content_final),
        safety_metadata   = _build_safety_metadata(proc, content_final),
    )


def _json_db_to_chunks(json_path: Path) -> List[ParsedChunk]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks: List[ParsedChunk] = []

    for proc in data:
        symptoms       = proc.get("symptoms", [])
        causes         = proc.get("causes", [])
        symptoms_text  = "\n".join(f"- {s}" for s in symptoms)
        causes_text    = "\n".join(f"- {c}" for c in causes)
        steps_text     = "\n".join(
            f"Step {s['step_number']}: {s['instruction']}"
            + (f"\n  → Expected: {s['expected_result']}" if s.get("expected_result") else "")
            + (f"\n  → If fail: {s['if_fail_then']}"    if s.get("if_fail_then")    else "")
            for s in proc.get("step_sequence", [])
        )
        warnings_text  = "\n".join(f"⚠️ {w}" for w in proc.get("safety_warnings", []))
        tools_text     = ", ".join(proc.get("required_tools", []))
        parts_text     = ", ".join(proc.get("part_numbers", []))
        symptoms_str   = ", ".join(symptoms)
        component      = proc.get("component", "")
        machine_family = proc.get("machine_family", "")
        system         = proc.get("system", "")
        machine_type   = proc.get("machine_type", "universal")
        escalate_if    = " | ".join(proc.get("safety_warnings", []))
        repair_refs    = proc.get("repair_refs", [])
        repair_refs_text = "\nRELATED REPAIR PROCEDURES:\n" + "\n".join(f"- {r}" for r in repair_refs) if repair_refs else ""
        source_file    = proc.get("manual_source", json_path.name)
        chunk_id       = proc.get("chunk_id") or hashlib.md5(f"{component}{symptoms_str}".encode()).hexdigest()[:12]
        tags           = sorted(set(t for t in (
            proc.get("symptoms_canonical", []) + [system.lower(), machine_type, component.lower()] + [r.lower() for r in repair_refs]
        ) if t))
        problem_categories = _infer_problem_categories(proc)
        variants_block = _synthetic_variants(symptoms, causes, component)

        proc_with_refs = dict(proc)
        proc_with_refs.setdefault("repair_refs", repair_refs)
        narrative = _generate_retrieval_narrative(proc_with_refs)

        # ── Chunk A: Narrative + overview ────────────────────────────────────
        overview_content = (
            f"PROBLEM: {component} — {symptoms_str}\n"
            f"MACHINE: {machine_family}\nSYSTEM: {system}\n"
            f"ESCALATE_IF: {escalate_if}\n\n"
            f"{narrative}\n\n"
            f"{variants_block}"
        )
        for i, sub in enumerate(split_with_overlap(overview_content)):
            cid = f"{chunk_id}_overview" if i == 0 else f"{chunk_id}_overview_{i}"
            chunks.append(ParsedChunk(
                chunk_id=cid, problem=f"{component} — {symptoms_str[:100]}",
                machine_type=machine_type, tags=tags,
                problem_categories=problem_categories,
                parts=[component] + proc.get("part_numbers", []),
                escalate_if=escalate_if, content=sub, source_file=source_file,
                has_repair_refs=bool(repair_refs),
                failure_taxonomy=_infer_failure_taxonomy(sub),
                safety_metadata=_build_safety_metadata(proc, sub),
            ))

        # ── Chunk B: Symptoms + causes ───────────────────────────────────────
        if symptoms_text or causes_text:
            sc_content = (
                f"PROBLEM: {component} — {symptoms_str}\n"
                f"MACHINE: {machine_family}\n\n"
                f"SYMPTOMS:\n{symptoms_text}\n\n"
                f"CAUSES:\n{causes_text}\n\n"
                f"{variants_block}"
            )
            for i, sub in enumerate(split_with_overlap(sc_content)):
                cid = f"{chunk_id}_symptoms" if i == 0 else f"{chunk_id}_symptoms_{i}"
                chunks.append(ParsedChunk(
                    chunk_id=cid, problem=f"{component} — {symptoms_str[:100]}",
                    machine_type=machine_type, tags=tags,
                    problem_categories=problem_categories,
                    parts=[component] + proc.get("part_numbers", []),
                    escalate_if=escalate_if, content=sub, source_file=source_file,
                    has_repair_refs=bool(repair_refs),
                    failure_taxonomy=_infer_failure_taxonomy(sub),
                    safety_metadata=_build_safety_metadata(proc, sub),
                ))

        # ── Chunk C: Repair steps ────────────────────────────────────────────
        if steps_text:
            steps_content = (
                f"REPAIR STEPS for {component} — {symptoms_str}\n"
                f"MACHINE: {machine_family}\n\n"
                f"{steps_text}\n\n"
                f"REQUIRED TOOLS: {tools_text}\nPART NUMBERS: {parts_text}\n"
                f"SOURCE: {source_file} | {proc.get('source_section', '')}"
                f"{repair_refs_text}"
            )
            for i, sub in enumerate(split_with_overlap(steps_content)):
                cid = f"{chunk_id}_steps" if i == 0 else f"{chunk_id}_steps_{i}"
                chunks.append(ParsedChunk(
                    chunk_id=cid, problem=f"{component} — {symptoms_str[:100]}",
                    machine_type=machine_type, tags=tags,
                    problem_categories=problem_categories,
                    parts=[component] + proc.get("part_numbers", []),
                    escalate_if=escalate_if, content=sub, source_file=source_file,
                    has_repair_refs=bool(repair_refs),
                    failure_taxonomy=_infer_failure_taxonomy(sub),
                    safety_metadata=_build_safety_metadata(proc, sub),
                ))

        # ── Chunk D: Safety + tools ──────────────────────────────────────────
        if warnings_text or tools_text:
            safety_content = (
                f"SAFETY & TOOLS for {component} — {symptoms_str}\n"
                f"MACHINE: {machine_family}\n\n"
                f"SAFETY WARNINGS:\n{warnings_text}\n\n"
                f"REQUIRED TOOLS: {tools_text}\nPART NUMBERS: {parts_text}\n"
                f"ESCALATE_IF: {escalate_if}\nSOURCE: {source_file}"
            )
            for i, sub in enumerate(split_with_overlap(safety_content)):
                cid = f"{chunk_id}_safety" if i == 0 else f"{chunk_id}_safety_{i}"
                chunks.append(ParsedChunk(
                    chunk_id=cid, problem=f"{component} — {symptoms_str[:100]}",
                    machine_type=machine_type, tags=tags,
                    problem_categories=problem_categories,
                    parts=[component] + proc.get("part_numbers", []),
                    escalate_if=escalate_if, content=sub, source_file=source_file,
                    has_repair_refs=bool(repair_refs),
                    failure_taxonomy=_infer_failure_taxonomy(sub),
                    safety_metadata=_build_safety_metadata(proc, sub),
                ))

    return chunks


def _fault_db_to_chunks(json_path: Path) -> List[ParsedChunk]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks: List[ParsedChunk] = []

    for fault in data:
        symptom      = fault.get("symptom", "")
        component    = fault.get("component", "")
        machine_fam  = fault.get("machine_family", "")
        machine_type = _resolve_machine_type(fault)
        source_file  = fault.get("manual_source", json_path.name)

        if not symptom: continue

        causes_text  = "\n".join(f"- {c}" for c in fault.get("likely_causes", []))
        verify_text  = "\n".join(f"- {v}" for v in fault.get("verify", []))
        repair_list  = fault.get("repair", []) if isinstance(fault.get("repair"), list) else ([fault.get("repair", "")] if fault.get("repair") else [])
        repair_text  = "\n".join(f"- {r}" for r in repair_list)
        repair_refs  = fault.get("repair_refs", [])
        repair_refs_text = "\nRELATED REPAIR PROCEDURES:\n" + "\n".join(f"- {r}" for r in repair_refs) if repair_refs else ""
        chunk_id     = "fault_" + hashlib.md5(f"{component}{symptom}".encode()).hexdigest()[:12]
        tags         = sorted(set(filter(None, [component.lower(), machine_type, symptom.lower()[:60]])))
        prob_cats    = _infer_problem_categories(fault)
        variants     = _synthetic_variants([symptom], fault.get("likely_causes", []), component)

        common_meta = dict(
            machine_type=machine_type, tags=tags, problem_categories=prob_cats,
            parts=[component] if component else [], escalate_if="",
            source_file=source_file, has_repair_refs=bool(repair_refs),
        )

        def _fault_chunk(cid, raw):
            raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
            for i, sub in enumerate(split_with_overlap(raw)):
                sub_id = cid if i == 0 else f"{cid}_{i}"
                chunks.append(ParsedChunk(
                    chunk_id=sub_id,
                    problem=f"{component} — {symptom[:100]}",
                    content=sub,
                    failure_taxonomy=_infer_failure_taxonomy(sub),
                    safety_metadata=_build_safety_metadata({}, sub),
                    **common_meta,
                ))

        # Chunk A: Fault overview
        _fault_chunk(f"{chunk_id}_overview", (
            f"FAULT: {symptom}\nCOMPONENT: {component}\nMACHINE: {machine_fam}\n"
            f"DESCRIPTION: {fault.get('problem_description', '')}\n"
            f"UNDERSTANDING: {fault.get('understanding', '')}\n\n"
            f"LIKELY CAUSES:\n{causes_text}\n\n{variants}"
        ))

        # Chunk B: Diagnostics + repair
        _fault_chunk(f"{chunk_id}_repair", (
            f"FAULT: {symptom}\nCOMPONENT: {component}\nMACHINE: {machine_fam}\n\n"
            f"CHECKS TO RUN:\n{verify_text}\n\n"
            f"REPAIR ACTIONS:\n{repair_text}"
            f"{repair_refs_text}\nSOURCE: {source_file} | {fault.get('source_section', '')}"
        ))

    return chunks


def _spec_db_to_chunks(json_path: Path) -> List[ParsedChunk]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks: List[ParsedChunk] = []

    for spec in data:
        component    = spec.get("component", "")
        parameter    = spec.get("parameter") or spec.get("spec_type", "")
        value        = spec.get("value", "")
        unit         = spec.get("unit", "")
        source_file  = spec.get("manual_source", json_path.name)
        machine_type = _resolve_machine_type(spec)

        if not component or not value: continue

        oor_text     = "\n".join(f"- {s}" for s in spec.get("if_out_of_range", []))
        repair_text  = "\n".join(f"- {r}" for r in spec.get("repair_actions", []))
        range_str    = spec.get("acceptable_range", "")

        content = (
            f"SPECIFICATION: {component} — {parameter}\n"
            f"VALUE: {value} {unit}"
            + (f"  (acceptable range: {range_str})" if range_str else "") + "\n"
            f"PAGE: {spec.get('page') or spec.get('source_page', '')}\n\n"
            + (f"IF OUT OF RANGE:\n{oor_text}\n\n" if oor_text else "")
            + (f"REPAIR ACTIONS:\n{repair_text}\n" if repair_text else "")
            + f"SOURCE: {source_file}"
        )
        content = re.sub(r"\n{3,}", "\n\n", content)

        chunk_id = "spec_" + hashlib.md5(f"{component}{parameter}{value}".encode()).hexdigest()[:12]
        tags = sorted(set(filter(None, [component.lower(), machine_type, parameter.lower()])))
        prob_cats = _infer_problem_categories({"system": parameter, "symptoms": spec.get("if_out_of_range", []), "causes": []})

        for i, sub in enumerate(split_with_overlap(content)):
            cid = chunk_id if i == 0 else f"{chunk_id}_{i}"
            chunks.append(ParsedChunk(
                chunk_id          = cid,
                problem           = f"{component} {parameter} = {value} {unit}",
                machine_type      = machine_type,
                tags              = tags,
                problem_categories= prob_cats,
                parts             = [component],
                escalate_if       = "",
                content           = sub,
                source_file       = source_file,
                has_repair_refs   = False,
                failure_taxonomy  = _infer_failure_taxonomy(sub),
                safety_metadata   = _build_safety_metadata({}, sub),
            ))

    return chunks


def _repair_db_to_chunks(json_path: Path) -> List[ParsedChunk]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks: List[ParsedChunk] = []

    for repair in data:
        procedure    = str(repair.get("procedure", "")).strip()
        component    = repair.get("component", "")
        machine_fam  = repair.get("machine_family", "")
        machine_type = _resolve_machine_type(repair)
        source_file  = repair.get("manual_source", json_path.name)

        if not procedure: continue

        steps        = repair.get("steps", [])
        steps_text   = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
        tools_text   = ", ".join(repair.get("tools", []))
        warnings_list= repair.get("safety_warnings", [])
        warnings_text= "\n".join(f"⚠️ {w}" for w in warnings_list)
        parts_text   = ", ".join(repair.get("part_numbers", []))
        chunk_id     = "repair_" + hashlib.md5(f"{component}{procedure}".encode()).hexdigest()[:12]
        tags         = sorted(set(filter(None, [component.lower(), machine_type, procedure.lower()[:60]])))
        prob_cats    = _infer_problem_categories({"system": component, "symptoms": [], "causes": steps})
        variants     = _synthetic_variants([], steps[:3], component)

        common_meta = dict(
            machine_type=machine_type, tags=tags, problem_categories=prob_cats,
            parts=[component] + repair.get("part_numbers", []),
            escalate_if=" | ".join(warnings_list),
            source_file=source_file, has_repair_refs=False,
        )

        def _repair_chunk(cid, raw):
            raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
            for i, sub in enumerate(split_with_overlap(raw)):
                sub_id = cid if i == 0 else f"{cid}_{i}"
                chunks.append(ParsedChunk(
                    chunk_id=sub_id,
                    problem=f"{procedure} — {component}",
                    content=sub,
                    failure_taxonomy=_infer_failure_taxonomy(sub),
                    safety_metadata=_build_safety_metadata({}, sub),
                    **common_meta,
                ))

        # Chunk A: Overview
        _repair_chunk(f"{chunk_id}_overview", (
            f"REPAIR PROCEDURE: {procedure}\nCOMPONENT: {component}\nMACHINE: {machine_fam}\n\n"
            f"REQUIRED TOOLS: {tools_text}\nPART NUMBERS: {parts_text}\n"
            f"EXPECTED RESULT: {repair.get('expected_result', '')}\n{variants}"
        ))

        # Chunk B: Steps + safety
        if steps_text or warnings_text:
            _repair_chunk(f"{chunk_id}_steps", (
                f"REPAIR PROCEDURE: {procedure}\nCOMPONENT: {component}\nMACHINE: {machine_fam}\n\n"
                f"STEPS:\n{steps_text}\n\n"
                f"SAFETY WARNINGS:\n{warnings_text}\n"
                f"SOURCE: {source_file} | {repair.get('source_section', '')}"
            ))

    return chunks


def _diagnostic_tree_to_chunks(json_path: Path) -> List[ParsedChunk]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks: List[ParsedChunk] = []

    for tree in data:
        symptom      = tree.get("symptom", "")
        component    = tree.get("component", "")
        machine_fam  = tree.get("machine_family", "")
        machine_type = _resolve_machine_type(tree)
        source_file  = tree.get("manual_source", json_path.name)

        for i, branch in enumerate(tree.get("branches", [])):
            question = branch.get("question", "")
            yes_path = branch.get("yes_next") or branch.get("yes", "")
            no_path  = branch.get("no_next")  or branch.get("no", "")

            if not question: continue

            content = (
                f"DIAGNOSTIC TREE — ROOT SYMPTOM: {symptom}\n"
                f"COMPONENT: {component}\n"
                f"MACHINE: {machine_fam}\n\n"
                f"QUESTION: {question}\n"
                f"IF YES: {yes_path}\n"
                f"IF NO: {no_path}\n"
                f"SOURCE: {source_file} | {tree.get('source_section', '')}"
            )
            content = re.sub(r"\n{3,}", "\n\n", content)

            chunk_id = "tree_" + hashlib.md5(f"{symptom}{question}".encode()).hexdigest()[:12]
            tags = sorted(set(filter(None, [component.lower(), machine_type, symptom.lower()[:60]])))
            prob_cats = _infer_problem_categories({"system": component, "symptoms": [symptom], "causes": [yes_path, no_path]})

            for j, sub in enumerate(split_with_overlap(content)):
                sub_id = chunk_id if j == 0 else f"{chunk_id}_{j}"
                chunks.append(ParsedChunk(
                    chunk_id          = sub_id,
                    problem           = f"{symptom[:80]} — {question[:60]}",
                    machine_type      = machine_type,
                    tags              = tags,
                    problem_categories= prob_cats,
                    parts             = [component] if component else [],
                    escalate_if       = "",
                    content           = sub,
                    source_file       = source_file,
                    has_repair_refs   = False,
                    failure_taxonomy  = _infer_failure_taxonomy(sub),
                    safety_metadata   = _build_safety_metadata({}, sub),
                ))

    return chunks

# ═══════════════════════════════════════════════════════════════════════════
# IMAGE DATABASE → ParsedChunk BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

_SEMANTIC_TOP_K = 3
_SEMANTIC_MIN_SIM = 0.15

def _compute_semantic_neighbours(records: List[dict]) -> List[dict]:
    texts = []
    for r in records:
        parts = [
            r.get("caption", ""),
            " ".join(_flatten_list(r.get("fault_relevance", []))),
            " ".join(_flatten_list(r.get("search_keywords", []))),
            r.get("section_context", ""),
            ", ".join(_flatten_list(r.get("components_shown", []))),
        ]
        texts.append(" ".join(p for p in parts if p).lower())

    n = len(texts)
    if n < 2:
        for r in records:
            r.setdefault("related_images", [])
        return records

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        import numpy as _np

        vec   = TfidfVectorizer(min_df=1, ngram_range=(1, 2), sublinear_tf=True)
        tfidf = vec.fit_transform(texts)
        sim   = _cos_sim(tfidf)

        for i, rec in enumerate(records):
            row      = sim[i]
            row[i]   = -1.0
            top_idxs = _np.argsort(row)[::-1][:_SEMANTIC_TOP_K]
            related  = []
            for j in top_idxs:
                if row[j] >= _SEMANTIC_MIN_SIM:
                    related.append({
                        "filename":   records[j]["filename"],
                        "similarity": round(float(row[j]), 3),
                        "caption":    records[j].get("caption", "")[:120],
                    })
            rec["related_images"] = related

    except ImportError:
        import math
        def _tokenize(t: str) -> List[str]:
            return re.findall(r"[a-z0-9]+", t.lower())
        def _tfidf_vec(tokens: List[str], idf: dict) -> dict:
            tf: dict = {}
            for tok in tokens: tf[tok] = tf.get(tok, 0) + 1
            total = max(len(tokens), 1)
            return {tok: (cnt / total) * idf.get(tok, 0) for tok, cnt in tf.items()}
        def _cosine(a: dict, b: dict) -> float:
            dot  = sum(a.get(t, 0) * b.get(t, 0) for t in a)
            norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
            return dot / norm if norm else 0.0

        token_lists = [_tokenize(t) for t in texts]
        df: dict = {}
        for tl in token_lists:
            for tok in set(tl): df[tok] = df.get(tok, 0) + 1
        idf = {tok: math.log(n / cnt + 1) for tok, cnt in df.items()}
        vecs = [_tfidf_vec(tl, idf) for tl in token_lists]

        for i, rec in enumerate(records):
            sims = [(j, _cosine(vecs[i], vecs[j])) for j in range(n) if j != i]
            sims.sort(key=lambda x: x[1], reverse=True)
            related = []
            for j, score in sims[:_SEMANTIC_TOP_K]:
                if score >= _SEMANTIC_MIN_SIM:
                    related.append({
                        "filename":   records[j]["filename"],
                        "similarity": round(score, 3),
                        "caption":    records[j].get("caption", "")[:120],
                    })
            rec["related_images"] = related

    logger.info("   🔗 Semantic neighbours computed for %d images", n)
    return records


# ═══════════════════════════════════════════════════════════════════════════
# FIX C: IMAGE DATABASE → ParsedChunk BRIDGE (Updated)
# ═══════════════════════════════════════════════════════════════════════════

def _image_db_to_chunks(json_path: Path) -> List[ParsedChunk]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    captioned = [img for img in data if img.get("caption")]
    if captioned:
        _compute_semantic_neighbours(captioned)
        captioned_set = {img["filename"]: img for img in captioned}
        data = [captioned_set.get(img.get("filename", ""), img) for img in data]

    chunks: List[ParsedChunk] = []

    for img in data:
        if not img.get("caption"): continue

        # FIX 1: Using _flatten_list helper for all potentially nested list fields
        keywords            = " ".join(_flatten_list(img.get("search_keywords", [])))
        components          = ", ".join(_flatten_list(img.get("components_shown", [])))
        figure_refs         = _flatten_list(img.get("figure_references", []))
        
        prev_ctx            = img.get("prev_context", "")
        next_ctx            = img.get("next_context", "")
        surrounding_text    = img.get("surrounding_text", "").strip()
        related_images      = img.get("related_images", [])
        diagram_type_raw    = img.get("diagram_type", "")
        diagram_type_canon  = img.get("diagram_type_classifier", "other")

        neighbour_block = ""
        if prev_ctx: neighbour_block += f"PREV_IMAGE: {prev_ctx}\n"
        if next_ctx: neighbour_block += f"NEXT_IMAGE: {next_ctx}\n"

        fig_ref_block = "FIGURE_REFS: " + ", ".join(figure_refs) + "\n" if figure_refs else ""

        related_block = ""
        if related_images:
            lines = []
            for r in related_images:
                if not isinstance(r, dict):
                    logger.warning(
                        "related_images entry is not a dict: %s — skipping",
                        type(r).__name__,
                    )
                    continue
                lines.append(
                    f"  - {r.get('filename', '?')} "
                    f"(sim={r.get('similarity', 0):.2f}): {r.get('caption', '')}"
                )
            related_block = "RELATED_IMAGES:\n" + "\n".join(lines) + "\n" if lines else ""

        surrounding_block = "PAGE_TEXT_CONTEXT:\n" + surrounding_text[:800] + "\n" if surrounding_text else ""

        content_raw = (
            f"DIAGRAM: {img.get('caption', '')}\n"
            f"DIAGRAM_TYPE: {diagram_type_raw}\n"
            f"DIAGRAM_TYPE_CLASSIFIER: {diagram_type_canon}\n"
            f"SECTION: {img.get('section_context', '')}\n"
            f"COMPONENTS: {components}\n"
            f"FAULT_RELEVANCE: {img.get('fault_relevance', '')}\n"
            f"KEYWORDS: {keywords}\n"
            f"{fig_ref_block}{neighbour_block}{related_block}{surrounding_block}"
            f"SOURCE: {img.get('manual_source', '')} | page {img.get('page', '')}\n"
            f"IMAGE_FILE: {img.get('filename', '')}"
        )
        content_raw = re.sub(r"\n{3,}", "\n\n", content_raw)

        source = img.get("manual_source", "")
        machine_type = _resolve_machine_type(img)
        chunk_id = "img_" + hashlib.md5(img.get("filename", img.get("caption", "")).encode()).hexdigest()[:12]

        tags = (
            _flatten_list(img.get("search_keywords", []))
            + _flatten_list([diagram_type_canon])
            + _flatten_list([diagram_type_raw])        # may be a list from CLIP pipelines
            + _flatten_list([img.get("section_context", "")])
            + _flatten_list(figure_refs)
        )
        tags = sorted(set(t.strip().lower() for t in tags if t and t.strip()))
        prob_cats = _infer_problem_categories({
            "system":   img.get("section_context", ""),
            "symptoms": _flatten_list(img.get("search_keywords", [])),
            "causes":   [],
        })
        fault_rel_raw = img.get("fault_relevance", "")
        if isinstance(fault_rel_raw, list):
            fault_rel_raw = " ".join(_flatten_list(fault_rel_raw))
        problem_str = (fault_rel_raw or img.get("caption", ""))[:120]

        for i, sub in enumerate(split_with_overlap(content_raw)):
            sub_id = chunk_id if i == 0 else f"{chunk_id}_{i}"
            chunks.append(ParsedChunk(
                chunk_id=sub_id,
                problem=problem_str,
                machine_type=machine_type,
                tags=tags,
                problem_categories=prob_cats,
                parts=_flatten_list(img.get("components_shown", [])),
                escalate_if="",
                content=sub,
                source_file=source,
            ))

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT GRAPH → ParsedChunk BRIDGE
# ═══════════════════════════════════════════════════════════════════════════
def _component_graph_to_chunks(json_path: Path) -> List[ParsedChunk]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks: List[ParsedChunk] = []

    for node in data:
        component = node.get("component", "")
        if not component: continue
        machine_type = _resolve_machine_type(node)

        connected = ", ".join(node.get("connected_to", []))
        faults = "\n- ".join(node.get("related_faults", []))
        specs = "\n- ".join(node.get("related_specs", []))
        aliases = ", ".join(node.get("aliases", []))

        content = f"COMPONENT GRAPH NODE: {component}\n"
        if aliases: content += f"ALIASES: {aliases}\n"
        content += f"MACHINE: {node.get('machine_family', '')}\n\n"
        content += f"PHYSICALLY CONNECTED TO: {connected}\n\n"
        if faults: content += f"RELATED FAULTS:\n- {faults}\n\n"
        if specs: content += f"RELATED SPECS:\n- {specs}\n"

        chunk_id = "graph_" + hashlib.md5(component.encode()).hexdigest()[:12]
        tags = sorted(set(filter(None, [component.lower(), machine_type])))

        for i, sub in enumerate(split_with_overlap(content)):
            sub_id = chunk_id if i == 0 else f"{chunk_id}_{i}"
            chunks.append(ParsedChunk(
                chunk_id=sub_id,
                problem=f"Graph Node: {component}",
                machine_type=machine_type,
                tags=tags,
                problem_categories=["general"],
                parts=[component] + node.get("connected_to", []),
                escalate_if="",
                content=sub,
                source_file=json_path.name,
                has_repair_refs=False,
                failure_taxonomy=_infer_failure_taxonomy(sub),
                safety_metadata=_build_safety_metadata({}, sub),
            ))
    return chunks

def _flatten_list(value) -> List[str]:
    """
    Safely flatten any combination of None / str / int / float / bool / nested list
    into a flat List[str].  Empty strings are dropped.
    """
    if value is None:
        return []
    if isinstance(value, (bool, int, float)):
        return [str(value)]
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_flatten_list(item))
        return out
    # fallback: any other type → str conversion
    return [str(value)]

# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE INDEX → ParsedChunk BRIDGE (Updated)
# ═══════════════════════════════════════════════════════════════════════════
def _knowledge_index_to_chunks(json_path: Path) -> List[ParsedChunk]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    chunks: List[ParsedChunk] = []

    for entry in data:
        component = entry.get("component", "")
        if not component: continue

        machine_type = _resolve_machine_type(entry)

        faults = "\n- ".join(_flatten_list(entry.get("fault_ids", [])))
        specs = "\n- ".join(_flatten_list(entry.get("spec_ids", [])))
        connected = ", ".join(_flatten_list(entry.get("connected_components", [])))

        content = f"KNOWLEDGE INDEX: {component}\n"
        content += f"CONNECTED COMPONENTS: {connected}\n\n"
        if faults: content += f"ASSOCIATED FAULTS:\n- {faults}\n\n"
        if specs: content += f"ASSOCIATED SPECS:\n- {specs}\n"

        chunk_id = "idx_" + hashlib.md5(component.encode()).hexdigest()[:12]
        tags = sorted(set(filter(None, [component.lower()])))

        for i, sub in enumerate(split_with_overlap(content)):
            sub_id = chunk_id if i == 0 else f"{chunk_id}_{i}"
            chunks.append(ParsedChunk(
                chunk_id=sub_id,
                problem=f"Knowledge Index: {component}",
                machine_type=machine_type,
                tags=tags,
                problem_categories=["general"],
                parts=[component] + _flatten_list(entry.get("connected_components", [])),
                escalate_if="",
                content=sub,
                source_file=json_path.name,
                has_repair_refs=False,
                failure_taxonomy=_infer_failure_taxonomy(sub),
                safety_metadata=_build_safety_metadata({}, sub),
            ))
    return chunks

# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeBaseBuilder:
    def __init__(self, knowledge_dir: str, db_dir: str):
        self.knowledge_dir = Path(knowledge_dir)
        self.db_dir = Path(db_dir)
        self.parser = StructuredChunkParser()
        self.stats = {
            "files_processed": 0, "files_failed": 0, "chunks_parsed": 0,
            "chunks_rejected": 0, "chunks_stored": 0, "machine_types": set(),
            "problem_categories": set(),
        }
    
    def build(self) -> bool:
        logger.info("=" * 80)
        logger.info("🚀 AgriFix RAG Knowledge Base Builder v5.3")
        logger.info("=" * 80)
        
        try:
            # We don't need Google API key for embeddings anymore, but we'll leave it in case 
            # other parts of the ecosystem expect it to exist in env.
            api_key = os.getenv("GOOGLE_AI_API_KEY", "placeholder") 
            
            all_chunks = self._parse_all_files()
            if not all_chunks:
                logger.error("❌ No valid chunks parsed from knowledge base!")
                return False
                
            documents = self._deduplicate([self._to_document(c) for c in all_chunks])
            success = self._embed_and_store(documents)
            self._generate_report(all_chunks)
            
            if success:
                logger.info("=" * 80)
                logger.info("✅ Knowledge base build completed successfully!")
                logger.info(f"📊 {self.stats['chunks_stored']} chunks stored in ChromaDB")
                logger.info(f"🤖 Machine types: {', '.join(self.stats['machine_types'])}")
                logger.info("=" * 80)
            return success
        except Exception as e:
            logger.error(f"❌ Build failed: {e}", exc_info=True)
            return False
            
    def _parse_all_files(self) -> List[ParsedChunk]:
        all_chunks: List[ParsedChunk] = []
        json_files = list(self.knowledge_dir.rglob("*.json"))

        image_files   = []
        fault_files   = []
        spec_files    = []
        repair_files  = []
        tree_files    = []
        graph_files   = []
        index_files   = []
        proc_files    = []   

        _SKIP_SUFFIXES = (
            "build_report.json",
        )

        _ENRICHED_SPECIAL_SUFFIXES = [
            "_fault_library_enriched.json",
            "_images_enriched.json",
            "_repair_procedures_enriched.json",
            "_component_graph_enriched.json",
            "_diagnostic_trees_enriched.json",
            "_knowledge_index_enriched.json",
            "_spec_database_enriched.json",
        ]

        # Collect all enriched stems to skip non-enriched duplicates
        all_names = {jf.name for jf in json_files}

        def _has_enriched_version(name: str) -> bool:
            """Return True if an enriched counterpart exists for this non-enriched file."""
            stem = re.sub(r"\.json$", "", name)
            enriched_name = stem + "_enriched.json"
            return enriched_name in all_names

        for jf in json_files:
            name = jf.name
            if any(name.endswith(s) or name == s for s in _SKIP_SUFFIXES):
                logger.debug("   ⏭  Skipping non-embeddable JSON: %s", name)
                continue
            # Skip original (non-enriched) if enriched version exists
            if not name.endswith("_enriched.json") and _has_enriched_version(name):
                logger.debug("   ⏭  Skipping non-enriched (enriched version present): %s", name)
                continue
            if name.endswith("_images_enriched.json") or (name.endswith("_images.json") and not _has_enriched_version(name)):
                image_files.append(jf)
            elif name.endswith("_fault_library_enriched.json") or (name.endswith("_fault_library.json") and not _has_enriched_version(name)):
                fault_files.append(jf)
            elif name.endswith("_spec_database_enriched.json") or (name.endswith("_spec_database.json") and not _has_enriched_version(name)):
                spec_files.append(jf)
            elif name.endswith("_repair_procedures_enriched.json") or (name.endswith("_repair_procedures.json") and not _has_enriched_version(name)):
                repair_files.append(jf)
            elif name.endswith("_diagnostic_trees_enriched.json") or (name.endswith("_diagnostic_trees.json") and not _has_enriched_version(name)):
                tree_files.append(jf)
            elif name.endswith("_component_graph_enriched.json") or (name.endswith("_component_graph.json") and not _has_enriched_version(name)):
                graph_files.append(jf)
            elif name.endswith("_knowledge_index_enriched.json") or (name.endswith("_knowledge_index.json") and not _has_enriched_version(name)):
                index_files.append(jf)
            elif name.endswith("_enriched.json") and not any(name.endswith(s) for s in _ENRICHED_SPECIAL_SUFFIXES):
                proc_files.append(jf)
            elif not name.endswith("_enriched.json"):
                # Non-enriched main DB — only load if no enriched version
                proc_files.append(jf)

        def _load(files, loader_fn, label):
            for jf in files:
                try:
                    chunks = loader_fn(jf)
                    all_chunks.extend(chunks)
                    self.stats["files_processed"] += 1
                    logger.info(f"   ✓ [{label}] {jf.name} → {len(chunks)} chunks")
                except Exception as e:
                    import traceback
                    logger.error(f"❌ Failed to load {label} {jf.name}: {e}")
                    logger.error(f"   Full traceback:\n{traceback.format_exc()}")
                    self.stats["files_failed"] += 1

        if proc_files:
            logger.info(f"📁 Found {len(proc_files)} procedure DB(s)")
            _load(proc_files, _json_db_to_chunks, "procedures")

        if fault_files:
            logger.info(f"📁 Found {len(fault_files)} fault library file(s)")
            _load(fault_files, _fault_db_to_chunks, "faults")

        if spec_files:
            logger.info(f"📁 Found {len(spec_files)} spec database file(s)")
            _load(spec_files, _spec_db_to_chunks, "specs")

        if repair_files:
            logger.info(f"📁 Found {len(repair_files)} repair procedure file(s)")
            _load(repair_files, _repair_db_to_chunks, "repairs")

        if tree_files:
            logger.info(f"📁 Found {len(tree_files)} diagnostic tree file(s)")
            _load(tree_files, _diagnostic_tree_to_chunks, "trees")

        if graph_files:
            logger.info(f"📁 Found {len(graph_files)} component graph file(s)")
            _load(graph_files, _component_graph_to_chunks, "graphs")

        if index_files:
            logger.info(f"📁 Found {len(index_files)} knowledge index file(s)")
            _load(index_files, _knowledge_index_to_chunks, "indices")

        if image_files:
            logger.info(f"🖼️  Found {len(image_files)} image DB(s)")
            _load(image_files, _image_db_to_chunks, "images")

        txt_files = list(self.knowledge_dir.rglob("*.txt"))
        if txt_files:
            logger.info(f"📁 Found {len(txt_files)} legacy .txt knowledge file(s)")
            for file_path in txt_files:
                try:
                    chunks = self.parser.parse_file(file_path)
                    all_chunks.extend(chunks)
                    self.stats["files_processed"] += 1
                except Exception as e:
                    logger.error(f"❌ Failed to process {file_path.name}: {e}")
                    self.stats["files_failed"] += 1

        if not all_chunks:
            logger.warning(
                f"⚠️  No knowledge files found in {self.knowledge_dir}. "
                "Copy your Master_*_DB.json files here or point knowledge_dir "
                "at the folder containing them."
            )

        self.stats["chunks_parsed"] = len(all_chunks)
        self.stats["chunks_rejected"] = len(self.parser.rejected_chunks)
        return all_chunks
        
    def _to_document(self, chunk: ParsedChunk) -> Document:
        self.stats["machine_types"].add(chunk.machine_type)
        self.stats["problem_categories"].update(chunk.problem_categories)
        return Document(page_content=chunk.content, metadata=chunk.to_metadata())
        
    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        seen_hashes: set = set()
        seen_content: set = set()
        unique = []
        for doc in documents:
            content_hash = doc.metadata.get("content_hash", "")
            # Normalize content for semantic dedup
            content_norm = re.sub(r"\s+", " ", doc.page_content).strip().lower()
            if content_hash and content_hash in seen_hashes:
                continue
            if content_norm in seen_content:
                continue
            if content_hash:
                seen_hashes.add(content_hash)
            seen_content.add(content_norm)
            unique.append(doc)
        removed = len(documents) - len(unique)
        if removed > 0:
            logger.info(f"🗑️  Removed {removed} duplicate chunks")
        return unique
        
    def _embed_and_store(self, documents: List[Document]) -> bool:
        import time
        logger.info("🧠 Creating vector database...")

        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        if self.db_dir.exists(): shutil.rmtree(self.db_dir)
        db = Chroma(persist_directory=str(self.db_dir), embedding_function=embeddings)
        
        batch_size = 64
        total = len(documents)
        total_batches = (total + batch_size - 1) // batch_size
        logger.info(f"📦 Embedding {total} chunks in {total_batches} batches")
        
        for batch_num, i in enumerate(range(0, total, batch_size), start=1):
            batch = documents[i:i + batch_size]
            try:
                db.add_documents(batch)
                logger.info(f"   ✅ Batch {batch_num}/{total_batches} ({i + len(batch)}/{total} chunks)")
            except Exception as e:
                logger.error(f"   ❌ Batch {batch_num} failed: {e} — retrying...")
                time.sleep(10)
                db.add_documents(batch)
            if batch_num < total_batches: time.sleep(2)
                
        self.stats["chunks_stored"] = len(db.get()["ids"])
        return True
        
    def _generate_report(self, chunks: List[ParsedChunk]):
        report = {
            "build_date": datetime.now().isoformat(),
            "statistics": {
                "chunks_parsed": self.stats["chunks_parsed"],
                "chunks_rejected": self.stats["chunks_rejected"],
                "chunks_stored": self.stats["chunks_stored"],
            },
            "rejected_chunks": [{"id": cid, "reason": r} for cid, r in self.parser.rejected_chunks]
        }
        report_path = self.db_dir / "build_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if self.parser.rejected_chunks:
            logger.warning(f"⚠️  {len(self.parser.rejected_chunks)} chunks rejected (see build_report.json)")

if __name__ == "__main__":
    builder = KnowledgeBaseBuilder(
        knowledge_dir=r"D:\AgriFix_Workspace\AgriFixAR_Python_Client\database_creation\water_pump_pdfs",
        db_dir="./chroma_db"
    )
    import sys
    sys.exit(0 if builder.build() else 1)