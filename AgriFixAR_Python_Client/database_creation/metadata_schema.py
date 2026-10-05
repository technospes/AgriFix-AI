from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional


# ── Field limits ───────────────────────────────────────────────────────────────
_MAX_PROBLEM_CHARS = 80
_MAX_TAG_CHARS     = 20
_MAX_TAGS          = 8
_MAX_PARTS         = 5
_MAX_PART_CHARS    = 30
_MAX_SOURCE_CHARS  = 80


# ── Component family taxonomy ─────────────────────────────────────────────────
_COMPONENT_FAMILY_MAP: Dict[str, str] = {
    # Rotating components
    "bearing":      "rotating_component",
    "shaft":        "rotating_component",
    "impeller":     "rotating_component",
    "flywheel":     "rotating_component",
    "pulley":       "rotating_component",
    "rotor":        "rotating_component",
    "fan":          "rotating_component",
    "coupling":     "rotating_component",
    "pto":          "rotating_component",
    "gear":         "rotating_component",
    # Electrical components
    "capacitor":    "electrical_component",
    "winding":      "electrical_component",
    "relay":        "electrical_component",
    "contactor":    "electrical_component",
    "mcb":          "electrical_component",
    "fuse":         "electrical_component",
    "terminal":     "electrical_component",
    "alternator":   "electrical_component",
    "battery":      "electrical_component",
    "starter":      "electrical_component",
    "solenoid":     "electrical_component",
    "sensor":       "electrical_component",
    "switch":       "electrical_component",
    # Hydraulic/fluid components
    "pump":         "hydraulic_component",
    "valve":        "hydraulic_component",
    "cylinder":     "hydraulic_component",
    "seal":         "hydraulic_component",
    "hose":         "hydraulic_component",
    "pipe":         "hydraulic_component",
    "filter":       "hydraulic_component",
    "injector":     "hydraulic_component",
    "nozzle":       "hydraulic_component",
    "foot valve":   "hydraulic_component",
    "primer":       "hydraulic_component",
    # Mechanical/structural components
    "belt":         "mechanical_component",
    "blade":        "mechanical_component",
    "tine":         "mechanical_component",
    "shear bolt":   "mechanical_component",
    "frame":        "mechanical_component",
    "clutch":       "mechanical_component",
    "brake":        "mechanical_component",
    "spring":       "mechanical_component",
    # Thermal/combustion components
    "piston":       "combustion_component",
    "cylinder head":"combustion_component",
    "glow plug":    "combustion_component",
    "injector pump":"combustion_component",
    "governor":     "combustion_component",
    "air filter":   "combustion_component",
    "fuel filter":  "combustion_component",
    "coolant":      "thermal_component",
    "radiator":     "thermal_component",
    "thermostat":   "thermal_component",
}

# ── Machine family taxonomy ───────────────────────────────────────────────────
_MACHINE_FAMILY_MAP: Dict[str, str] = {
    "tractor":          "agricultural_vehicle",
    "harvester":        "agricultural_vehicle",
    "rotavator":        "tillage_implement",
    "power_tiller":     "agricultural_vehicle",
    "chaff_cutter":     "processing_machine",
    "thresher":         "processing_machine",
    "water_pump":       "irrigation_machine",
    "submersible_pump": "irrigation_machine",
    "electric_motor":   "electric_machine",
    "diesel_engine":    "prime_mover",
    "generator":        "electric_machine",
}

# ── Risk level rules ──────────────────────────────────────────────────────────
_RISK_KEYWORDS: Dict[str, List[str]] = {
    "critical": [
        "live wire", "high voltage", "electric shock", "electrocution",
        "fire hazard", "explosion", "fuel leak", "lockout", "do not operate",
        "rotating blades", "pto shaft", "chaff cutter blade",
    ],
    "high": [
        "capacitor", "mcb", "terminal", "winding", "hydraulic pressure",
        "hot surface", "burn", "overheating", "breaker", "electrical",
        "motor cover", "short circuit", "earth fault", "single phasing",
    ],
    "medium": [
        "belt", "impeller", "bearing", "seal", "valve", "pump casing",
        "fuel filter", "injector", "governor", "oil leak",
    ],
    "low": [
        "clean", "inspect", "check level", "visual inspection",
        "tighten", "lubricate", "air filter",
    ],
}

# ── Fault severity rules ──────────────────────────────────────────────────────
_SEVERITY_KEYWORDS: Dict[str, List[str]] = {
    "critical": [
        "burnt", "broken shaft", "seized", "melted", "explosion", "fire",
        "complete failure", "cannot start", "no discharge", "pump seized",
    ],
    "major": [
        "overheating", "tripped", "bearing failure", "belt broken",
        "impeller blocked", "no water", "single phase", "capacitor failed",
        "voltage drop", "overload", "shaft bent",
    ],
    "minor": [
        "vibration", "noise", "low pressure", "slow start", "reduced flow",
        "minor leak", "belt slip", "loose connection",
    ],
}


def infer_component_family(component: str, tags: List[str] = None) -> str:
    """Auto-infer component_family from component name and tags."""
    if not component and not tags:
        return "general_component"
    search_text = (component or "").lower()
    if tags:
        search_text += " " + " ".join(t.lower() for t in tags)
    for keyword, family in _COMPONENT_FAMILY_MAP.items():
        if keyword in search_text:
            return family
    return "general_component"


def infer_machine_family(machine_type: str) -> str:
    """Auto-infer machine_family from machine_type."""
    return _MACHINE_FAMILY_MAP.get(
        (machine_type or "").lower().strip(), "general_machine"
    )


def infer_risk_level(
    component: str = "",
    tags: List[str] = None,
    failure_taxonomy: List[str] = None,
    existing_risk: str = "",
) -> str:
    """Auto-infer risk_level from component, tags, and taxonomy."""
    if existing_risk and existing_risk.upper() in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        return existing_risk.upper()
    search_text = " ".join(filter(None, [
        (component or "").lower(),
        " ".join(t.lower() for t in (tags or [])),
        " ".join(t.lower() for t in (failure_taxonomy or [])),
    ]))
    for level in ("critical", "high", "medium", "low"):
        if any(kw in search_text for kw in _RISK_KEYWORDS[level]):
            return level.upper()
    return "MEDIUM"


def infer_fault_severity(
    symptoms: List[str] = None,
    causes: List[str] = None,
    existing_severity: str = "",
) -> str:
    """Auto-infer fault_severity from symptoms and causes."""
    if existing_severity and existing_severity.lower() in ("minor", "major", "critical"):
        return existing_severity.lower()
    search_text = " ".join(filter(None, [
        " ".join((symptoms or [])),
        " ".join((causes or [])),
    ])).lower()
    for level in ("critical", "major", "minor"):
        if any(kw in search_text for kw in _SEVERITY_KEYWORDS[level]):
            return level
    return "major"


# ── Audit helper ───────────────────────────────────────────────────────────────

def audit_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Audit a metadata dict for ChromaDB compliance. Returns violation list."""
    violations = []
    for key, value in metadata.items():
        if isinstance(value, str):
            if len(value) > 150:
                violations.append(
                    f"Field '{key}' is {len(value)} chars (max 150). "
                    f"Preview: '{value[:60]}...'"
                )
        elif isinstance(value, list):
            total_chars = sum(len(str(v)) for v in value)
            if total_chars > 400:
                violations.append(
                    f"Field '{key}' list totals {total_chars} chars across {len(value)} items."
                )
            for item in value:
                if isinstance(item, str) and len(item) > 40:
                    violations.append(
                        f"Field '{key}' has long item ({len(item)} chars): '{item[:40]}...'"
                    )
        elif isinstance(value, (int, float, bool)):
            pass
        elif value is None:
            pass
        else:
            violations.append(
                f"Field '{key}' has unsupported type {type(value).__name__}."
            )
    return violations


# ── Main metadata builder ──────────────────────────────────────────────────────

def clean_to_metadata(chunk) -> Dict[str, Any]:
    """
    Corrected to_metadata() for ParsedChunk.
    Adds: component_family, machine_family, risk_level (normalised), fault_severity.
    Backward-compatible: all existing field names preserved.
    escalate_if removed from metadata (in page_content instead).
    """
    content_hash = hashlib.md5(
        re.sub(r"\s+", " ", chunk.content).strip().encode()
    ).hexdigest()

    problem_short = (chunk.problem or "")[:_MAX_PROBLEM_CHARS].strip()

    tags_clean = _trim_list(
        chunk.tags if chunk.tags else ["none"],
        max_items=_MAX_TAGS,
        max_item_chars=_MAX_TAG_CHARS,
    )

    problem_categories = (
        chunk.problem_categories[:_MAX_TAGS]
        if chunk.problem_categories
        else ["general"]
    )

    parts_clean = _trim_list(
        chunk.parts if chunk.parts else ["none"],
        max_items=_MAX_PARTS,
        max_item_chars=_MAX_PART_CHARS,
    )

    source_file = (chunk.source_file or "")
    if "/" in source_file or "\\" in source_file:
        source_file = source_file.replace("\\", "/").split("/")[-1]
    source_file = source_file[:_MAX_SOURCE_CHARS]

    failure_taxonomy = (
        chunk.failure_taxonomy[:_MAX_TAGS]
        if chunk.failure_taxonomy
        else ["mechanical"]
    )

    sm = chunk.safety_metadata or {}

    # ── Normalised taxonomy fields ────────────────────────────────────────────
    component     = getattr(chunk, "component", "") or ""
    machine_type  = str(chunk.machine_type).lower()[:30]
    symptoms      = getattr(chunk, "symptoms", []) or []
    causes        = getattr(chunk, "causes", []) or []

    component_family = infer_component_family(component, tags_clean)
    machine_family   = infer_machine_family(machine_type)
    risk_level       = infer_risk_level(
        component        = component,
        tags             = tags_clean,
        failure_taxonomy = failure_taxonomy,
        existing_risk    = str(sm.get("risk_level", "")),
    )
    fault_severity   = infer_fault_severity(
        symptoms          = symptoms,
        causes            = causes,
        existing_severity = getattr(chunk, "severity", ""),
    )

    metadata: Dict[str, Any] = {
        # Identification
        "chunk_id":           str(chunk.chunk_id)[:40],
        "content_hash":       content_hash,
        "source_file":        source_file,
        # Machine targeting
        "machine_type":       machine_type,
        "machine_family":     machine_family[:30],
        # Component targeting
        "component_family":   component_family[:30],
        # Problem identifier
        "problem":            problem_short,
        # Taxonomy
        "failure_taxonomy":   failure_taxonomy,
        "problem_categories": problem_categories,
        "tags":               tags_clean,
        # Parts list
        "parts":              parts_clean,
        # Normalised risk + severity
        "risk_level":         risk_level[:12],
        "fault_severity":     fault_severity[:12],
        # Safety flags
        "electrical_hazard":  bool(sm.get("electrical_hazard", False)),
        "shutdown_required":  bool(sm.get("shutdown_required", False)),
        "expert_required":    bool(sm.get("expert_required", False)),
        # escalate_if intentionally NOT here — in page_content
    }

    violations = audit_metadata(metadata)
    if violations:
        import logging
        _logger = logging.getLogger(__name__)
        for v in violations:
            _logger.warning("Metadata violation [%s]: %s", chunk.chunk_id, v)

    return metadata


def _trim_list(items: List, max_items: int, max_item_chars: int) -> List[str]:
    result = []
    seen: set = set()
    for item in items:
        s = str(item).strip()[:max_item_chars]
        if s and s not in seen:
            seen.add(s)
            result.append(s)
        if len(result) >= max_items:
            break
    return result if result else ["none"]


def extract_escalate_if_from_content(page_content: str) -> str:
    """Extract ESCALATE_IF line from page_content. Use in rag.py instead of metadata."""
    match = re.search(r"ESCALATE_IF:\s*(.+?)(?:\n|$)", page_content, re.IGNORECASE)
    if match:
        return match.group(1).strip()[:200]
    return ""


def build_normalized_metadata(
    chunk_id: str,
    machine_type: str,
    component: str = "",
    symptoms: List[str] = None,
    causes: List[str] = None,
    tags: List[str] = None,
    failure_taxonomy: List[str] = None,
    risk_level: str = "",
    fault_severity: str = "",
    electrical_hazard: bool = False,
    shutdown_required: bool = False,
    expert_required: bool = False,
    problem: str = "",
    parts: List[str] = None,
    source_file: str = "",
    content: str = "",
) -> Dict[str, Any]:
    """
    Build fully-normalised metadata dict from raw fields.
    Use this when constructing metadata outside ParsedChunk (e.g. tests, scripts).
    All taxonomy fields are auto-populated if not explicitly provided.
    """
    tags_clean = _trim_list(tags or ["none"], _MAX_TAGS, _MAX_TAG_CHARS)
    ft = _trim_list(failure_taxonomy or ["mechanical"], _MAX_TAGS, _MAX_TAG_CHARS)

    norm_risk     = infer_risk_level(component, tags_clean, ft, risk_level)
    norm_severity = infer_fault_severity(symptoms, causes, fault_severity)
    comp_family   = infer_component_family(component, tags_clean)
    mach_family   = infer_machine_family(machine_type)

    content_hash = hashlib.md5(
        re.sub(r"\s+", " ", content).strip().encode()
    ).hexdigest() if content else ""

    sf = (source_file or "")
    if "/" in sf or "\\" in sf:
        sf = sf.replace("\\", "/").split("/")[-1]

    return {
        "chunk_id":           str(chunk_id)[:40],
        "content_hash":       content_hash,
        "source_file":        sf[:_MAX_SOURCE_CHARS],
        "machine_type":       str(machine_type).lower()[:30],
        "machine_family":     mach_family[:30],
        "component_family":   comp_family[:30],
        "problem":            (problem or "")[:_MAX_PROBLEM_CHARS],
        "failure_taxonomy":   ft,
        "problem_categories": ["general"],
        "tags":               tags_clean,
        "parts":              _trim_list(parts or ["none"], _MAX_PARTS, _MAX_PART_CHARS),
        "risk_level":         norm_risk[:12],
        "fault_severity":     norm_severity[:12],
        "electrical_hazard":  electrical_hazard,
        "shutdown_required":  shutdown_required,
        "expert_required":    expert_required,
    }


def audit_knowledge_base_json(json_path: str) -> None:
    """Audit all procedures in a Master_*_DB.json for metadata compliance."""
    import json, logging
    _logger = logging.getLogger(__name__)
    data = json.loads(open(json_path, encoding="utf-8").read())
    print(f"\nAuditing {json_path} ({len(data)} procedures)...\n")

    field_violations: Dict[str, list] = {}
    for i, proc in enumerate(data):
        escalate_if = " | ".join(proc.get("safety_warnings", []))
        problem     = proc.get("component", "?") + " — " + ", ".join(proc.get("symptoms", []))
        checks = {
            "escalate_if": escalate_if,
            "problem":     problem,
        }
        for field, value in checks.items():
            if isinstance(value, str) and len(value) > 150:
                if field not in field_violations:
                    field_violations[field] = []
                field_violations[field].append({
                    "proc_idx": i,
                    "chunk_id": proc.get("chunk_id", f"proc_{i}"),
                    "len":      len(value),
                    "preview":  value[:80],
                })

    for field, violations in field_violations.items():
        print(f"Field '{field}': {len(violations)} violations")
        for v in violations[:3]:
            print(f"  [{v['chunk_id']}] {v['len']} chars: {v['preview']}...")
        if len(violations) > 3:
            print(f"  ... and {len(violations) - 3} more")
        print()

    if not field_violations:
        print("No metadata violations found.")
    else:
        print(f"ACTION: Apply clean_to_metadata() to fix violations.")


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging, sys
    logging.basicConfig(level=logging.WARNING)

    print("=== Component Family Inference ===")
    for comp in ["capacitor", "bearing", "impeller", "relay", "fuel filter", "belt"]:
        print(f"  {comp!r:20} → {infer_component_family(comp)}")

    print("\n=== Machine Family Inference ===")
    for mt in ["tractor", "water_pump", "electric_motor", "thresher", "generator"]:
        print(f"  {mt!r:20} → {infer_machine_family(mt)}")

    print("\n=== Risk Level Inference ===")
    cases = [
        ("capacitor", ["electrical"], ["electrical_component"], ""),
        ("bearing",   ["mechanical"], ["rotating_component"],   ""),
        ("belt",      ["mechanical"], [],                        ""),
        ("",          [],             [],                        "CRITICAL"),
    ]
    for comp, tags, taxonomy, existing in cases:
        result = infer_risk_level(comp, tags, taxonomy, existing)
        print(f"  comp={comp!r:15} existing={existing!r:10} → {result}")

    print("\n=== Fault Severity Inference ===")
    cases2 = [
        (["motor burnt", "seized"], [], ""),
        (["no water flow"],         ["capacitor failed"], ""),
        (["minor vibration"],       [], ""),
    ]
    for symptoms, causes, existing in cases2:
        result = infer_fault_severity(symptoms, causes, existing)
        print(f"  symptoms={str(symptoms)[:35]:35} → {result}")

    print("\n=== build_normalized_metadata ===")
    m = build_normalized_metadata(
        chunk_id="test_001",
        machine_type="water_pump",
        component="capacitor",
        symptoms=["motor hums but does not rotate"],
        causes=["failed start capacitor"],
        tags=["capacitor", "not_starting", "humming"],
        failure_taxonomy=["electrical"],
        electrical_hazard=True,
        shutdown_required=True,
        problem="Capacitor — motor humming not starting",
        source_file="kirloskar_manual.pdf",
    )
    for k, v in m.items():
        print(f"  {k:25}: {v}")

    violations = audit_metadata(m)
    print(f"\nAudit violations: {violations or 'None'}")

    if len(sys.argv) > 1:
        audit_knowledge_base_json(sys.argv[1])