from __future__ import annotations
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from utils.groq_client import groq_client, TEXT_MODEL, groq_chat_completion

logger = logging.getLogger(__name__)

_MACHINE_REGISTRY: Dict[str, Dict] = {}
KNOWN_MACHINE_IDS: Set[str] = set()
_MACHINE_ALIAS_MAP: Dict[str, str] = {}
_EXPANSION_BANK: Dict[str, List[str]] = {}  # machine|symptom → [variants]  (populated dynamically)

def _extract_components_from_query(query: str, machine_type: str) -> List[str]:
    """
    Extract component/part names from the user's query by matching against
    the machine registry's part names using overlap scoring.
    
    A part matches if >= 50% of its tokens appear in the query.
    Results are ranked by overlap score, capped at 3.
    Avoids false positives from single-token matches to generic words.
    """
    q = query.lower()
    q_words = set(w for w in q.split() if len(w) > 2)
    
    from utils.machine_registry import get_profile, get_all_part_ids
    
    scored: List[tuple] = []  # (part_id, overlap_score)
    
    # Try machine-specific parts first
    profile = get_profile(machine_type)
    part_ids_to_check = []
    if profile:
        part_ids_to_check = [p.id for p in profile.parts]
    else:
        part_ids_to_check = get_all_part_ids(None)
    
    for part_id in part_ids_to_check:
        part_tokens = [t for t in part_id.replace("_", " ").split() if len(t) > 2]
        if not part_tokens:
            continue
        
        overlap = sum(1 for t in part_tokens if re.search(rf'\b{re.escape(t)}\b', q))
        score = overlap / len(part_tokens)
        
        if score >= 0.5:  # At least half the tokens match
            scored.append((part_id, score))
    
    # Sort by score descending, then by part_id for stability
    scored.sort(key=lambda x: (-x[1], x[0]))
    
    # Take top 3, dedup
    result = list(dict.fromkeys(p[0] for p in scored))[:3]
    
    if result:
        logger.debug("Component extraction: query='%s' machine=%s → %s (scores: %s)",
                     query[:60], machine_type, result,
                     [(p[0], f"{p[1]:.2f}") for p in scored[:3]])
    
    return result

def load_machine_registry(vector_db=None) -> None:
    global _MACHINE_REGISTRY, KNOWN_MACHINE_IDS, _MACHINE_ALIAS_MAP, _EXPANSION_BANK

    registry: Dict[str, Dict] = {}

    if vector_db is not None:
        try:
            # Pull all metadata from ChromaDB (metadatas only, no embeddings)
            collection = vector_db._collection
            result = collection.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []

            for meta in metadatas:
                if not meta:
                    continue

                machine_id = str(meta.get("machine_type", "")).lower().strip()
                if not machine_id or machine_id in ("unknown", "universal", ""):
                    continue

                if machine_id not in registry:
                    registry[machine_id] = {"aliases": set(), "symptoms": set()}

                # Parse aliases
                raw_aliases = meta.get("machine_alias", "") or meta.get("aliases", "")
                for alias in _parse_list_field(raw_aliases):
                    registry[machine_id]["aliases"].add(alias.lower().strip())

                # Parse symptoms
                raw_symptoms = meta.get("symptoms", "") or meta.get("problem", "")
                for symptom in _parse_list_field(raw_symptoms):
                    registry[machine_id]["symptoms"].add(symptom.lower().strip())

            logger.info(
                "Machine registry loaded from ChromaDB: %d machines → %s",
                len(registry), list(registry.keys()),
            )

        except Exception as exc:
            logger.warning("Could not load machine registry from ChromaDB: %s", exc)

    # If DB gave nothing (empty DB, first run, or test env), registry stays {}
    # The router will still work — machine_type will be "unknown" for all queries
    # and the LLM path will use whatever machine list it was last fine-tuned on.

    # Freeze sets → lists for JSON-serializability
    for mid, data in registry.items():
        data["aliases"] = list(data["aliases"])
        data["symptoms"] = list(data["symptoms"])

    _MACHINE_REGISTRY = registry

    # Flat set of known IDs (used for validation in _parse_router_json)
    KNOWN_MACHINE_IDS = set(registry.keys())

    # Alias map: every alias token → canonical machine_id (used in keyword fallback)
    alias_map: Dict[str, str] = {}
    for machine_id, data in registry.items():
        # The machine_id itself is always a valid alias
        alias_map[machine_id] = machine_id
        alias_map[machine_id.replace("_", " ")] = machine_id
        for alias in data["aliases"]:
            alias_map[alias] = machine_id
    _MACHINE_ALIAS_MAP = alias_map

    # Build expansion bank from (machine × symptoms) pairs found in the DB
    _EXPANSION_BANK = _build_expansion_bank(registry)

    logger.info(
        "Registry ready: %d machines, %d alias tokens, %d expansion entries",
        len(KNOWN_MACHINE_IDS), len(_MACHINE_ALIAS_MAP), len(_EXPANSION_BANK),
    )


def _parse_list_field(value) -> List[str]:
    """
    Normalise a metadata field that may be a plain string, comma-separated
    string, or JSON-encoded list into a Python list of strings.
    """
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    s = str(value).strip()
    # Try JSON array first
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except json.JSONDecodeError:
            pass
    # Comma-separated fallback
    return [p.strip() for p in s.split(",") if p.strip()]


def _build_expansion_bank(registry: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Build the expansion bank dynamically from the registry.

    For each (machine, symptom) pair seen in the DB, generate a small set of
    search variants by combining the machine name, its aliases, and the symptom
    in different orderings.  This replaces the old hardcoded _EXPANSION_BANK.

    The LLM will always generate better contextual variants at query time; this
    bank is only the static fallback supplement used when the LLM is unavailable.
    """
    bank: Dict[str, List[str]] = {}

    for machine_id, data in registry.items():
        machine_label = machine_id.replace("_", " ")
        aliases = data["aliases"][:3]   # top 3 aliases max, keep variants short

        for symptom in data["symptoms"]:
            key = f"{machine_id}|{symptom}"
            variants: List[str] = []

            # Pattern 1: "machine_label symptom"
            variants.append(f"{machine_label} {symptom}")

            # Pattern 2: alias + symptom (use first short alias if available)
            short_aliases = [a for a in aliases if len(a.split()) <= 2]
            if short_aliases:
                variants.append(f"{short_aliases[0]} {symptom}")

            # Pattern 3: symptom + machine_label (inverted, helps semantic search)
            variants.append(f"{symptom} in {machine_label}")

            # Pattern 4: "machine_label not working" / "machine_label problem"
            # (broad fallback variant for vague queries)
            if "not start" in symptom or "dead" in symptom:
                variants.append(f"{machine_label} not working")
                # FIX 4 sub-fix 3: emit intermittent cold-start variant so that
                # Hinglish temporal queries ("subah start nahi") retrieve the
                # correct intermittent-fault chunks instead of scoring 0.0.
                variants.append(f"{machine_label} intermittent not starting cold weather")
            elif "overheat" in symptom or "hot" in symptom or "garam" in symptom:
                variants.append(f"{machine_label} temperature high")
            elif "vibrat" in symptom or "shake" in symptom:
                variants.append(f"{machine_label} abnormal vibration")

            bank[key] = list(dict.fromkeys(variants))[:5]   # dedup, cap at 5

    return bank


# ── LLM prompt — fully dynamic machine list ───────────────────────────────────

def _build_router_prompt(query: str) -> str:
    """
    Build the router/expansion prompt with the machine list injected dynamically
    from whatever is in KNOWN_MACHINE_IDS (loaded from the DB at startup).

    If the registry is empty (cold start / test), the prompt tells the LLM to
    use its own judgment and return "unknown" if unsure.
    """
    if KNOWN_MACHINE_IDS:
        machine_list_str = " | ".join(sorted(KNOWN_MACHINE_IDS))
        machine_instruction = (
            f'machine_type: Pick ONE from this exact list or return "unknown":\n'
            f'    {machine_list_str}'
        )
    else:
        machine_instruction = (
            'machine_type: Identify the machine from the query using your own knowledge,\n'
            '    or return "unknown" if the machine cannot be identified.'
        )

    return f"""\
You are an expert diagnostic router for agricultural machinery.
Read the user's query (may be Hindi, English, or mixed Hinglish).
Return ONLY valid JSON — no markdown, no preamble.

TASK 1 — EXTRACT:
  {machine_instruction}
  symptoms: List of SHORT English phrases (max 5). 
  CRITICAL: If the user query is in Hindi or Hinglish (e.g. "paani nahi de raha", "awaaz karta hai"), you MUST translate the symptoms into standard English mechanical terms (e.g. "no water discharge", "making noise"). Do not return Hindi words in the symptoms array.
  confidence: Float 0.0–1.0 for machine_type certainty.
  language: "hi" | "en" | "mixed"

TASK 2 — EXPAND (3–5 search variants):
  Generate 3–5 alternative search queries that capture the same fault
  from different angles. Rules:
  • Keep each variant to 4–8 words.
  • Cover different likely root causes.
  • Mix technical terms with plain language.
  • Include at least 1 Hinglish variant if original query is Hindi/mixed.
  • Do NOT repeat the original query verbatim.
  • FIX 4: If the query contains time-of-day or frequency cues (morning,
    subah, raat, sometimes, kabhi kabhi, occasionally, thodi der baad,
    after long use), include a variant with "intermittent fault" in it
    (e.g. "electric motor intermittent not starting cold start").

Examples:
  Input: "motor awaz kar raha hai"
  Variants: ["motor humming not starting", "motor buzzing no rotation",
             "machine vibration abnormal noise", "electric motor not starting",
             "motor gharrr awaaz start nahi"]

  Input: "pump no water"
  Variants: ["pump not pumping water", "pump no discharge flow",
             "air lock suction pump", "impeller blocked pump",
             "pump chal raha paani nahi aa raha"]

Return EXACTLY this JSON:
{{
  "machine_type": "<id or unknown>",
  "symptoms": ["<symptom 1>", "<symptom 2>"],
  "confidence": 0.0,
  "language": "<hi|en|mixed>",
  "query_variants": ["<variant 1>", "<variant 2>", "<variant 3>"]
}}

Query: {query}
"""


# ── Output model ───────────────────────────────────────────────────────────────

@dataclass
class RouterOutput:
    """Structured output from the query router."""
    machine_type: str                    # one of KNOWN_MACHINE_IDS or "unknown"
    symptoms: List[str]                  # English symptom phrases
    confidence: float                    # 0.0–1.0 for machine_type
    language: str                        # "hi" | "en" | "mixed"
    raw_query: str = ""                       # original farmer input, unmodified
    query_variants: List[str] = field(default_factory=list)
    router_ok: bool = True               # False if router call failed
    error: Optional[str] = None         # populated on failure


# ── Internal helpers ───────────────────────────────────────────────────────────

def _parse_router_json(raw: str, query: str) -> RouterOutput:
    """Parse LLM response into RouterOutput; degrade gracefully on error."""
    try:
        from utils.json_repair import repair_json
        data = repair_json(raw)
        machine = str(data.get("machine_type", "unknown")).lower().strip()

        # Validate against dynamic registry; accept if registry is empty (cold start)
        if KNOWN_MACHINE_IDS and machine not in KNOWN_MACHINE_IDS:
            # Try alias resolution before rejecting
            resolved = _MACHINE_ALIAS_MAP.get(machine)
            if resolved:
                logger.info("Router: resolved alias '%s' → '%s'", machine, resolved)
                machine = resolved
            else:
                logger.warning("Router: unknown machine_type '%s' → 'unknown'", machine)
                machine = "unknown"

        symptoms = data.get("symptoms", [])
        if not isinstance(symptoms, list):
            symptoms = [str(symptoms)]
        symptoms = [str(s).strip() for s in symptoms if str(s).strip()][:5]

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        lang = str(data.get("language", "en")).lower()
        if lang not in ("hi", "en", "mixed"):
            lang = "en"

        variants_raw = data.get("query_variants", [])
        if not isinstance(variants_raw, list):
            variants_raw = []
        variants = [str(v).strip() for v in variants_raw if str(v).strip()][:5]

        # Supplement with dynamic bank variants
        bank_variants = _lookup_expansion_bank(machine, symptoms)
        all_variants = list(dict.fromkeys(variants + bank_variants))[:5]

        logger.info(
            "Router OK: machine=%s conf=%.2f symptoms=%s variants=%d lang=%s",
            machine, confidence, symptoms, len(all_variants), lang,
        )
        return RouterOutput(
            machine_type=machine,
            symptoms=symptoms,
            confidence=confidence,
            language=lang,
            raw_query=query,
            query_variants=all_variants,
        )

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error("Router parse error: %s | raw=%s", exc, raw[:200])
        return RouterOutput(
            machine_type="unknown",
            symptoms=[],
            confidence=0.0,
            language="en",
            raw_query=query,
            query_variants=[],
            router_ok=False,
            error=f"Parse error: {exc}",
        )

def _lookup_expansion_bank(machine_type: str, symptoms: List[str]) -> List[str]:
    """
    Look up pre-built query variants from the dynamically generated expansion bank.
    Returns up to 3 variants to supplement LLM-generated ones.
    """
    if machine_type == "unknown" or not symptoms:
        return []

    symptom_text = " ".join(s.lower() for s in symptoms)
    results: List[str] = []

    for key, variants in _EXPANSION_BANK.items():
        machine_key, symptom_kw = key.split("|", 1)
        if machine_key == machine_type and symptom_kw in symptom_text:
            results.extend(variants)
            break  # one bank hit is enough

    return results[:3]


# ── Keyword fallback ───────────────────────────────────────────────────────────

def _keyword_fallback(query: str) -> RouterOutput:
    """
    Deterministic fallback when the LLM router is unavailable.

    Machine detection uses _MACHINE_ALIAS_MAP which is built dynamically from
    the registry — no hardcoded machine names or if/elif chains.

    Symptom detection uses a language-level keyword map (Hindi + English surface
    forms for universal fault categories). These are linguistic patterns, NOT
    machine-specific logic, so they require no updates when machines change.
    """
    q = query.lower()

    # ── Machine detection via dynamic alias map ──────────────────────────────
    # Score each known machine by how many of its alias tokens appear in the query.
    # The machine with the highest overlap wins.
    machine_type = "unknown"

    if _MACHINE_ALIAS_MAP:
        scores: Dict[str, int] = {}
        for alias_token, machine_id in _MACHINE_ALIAS_MAP.items():
            # BUG 1 FIX: word-boundary match so "tractor" doesn't fire inside
            # "contractor", "pump" doesn't fire inside "pumped out water", etc.
            if re.search(rf'\b{re.escape(alias_token)}\b', q):
                scores[machine_id] = scores.get(machine_id, 0) + 1

        if scores:
            # Pick the machine with the most alias hits; tie → first alphabetically
            machine_type = max(scores, key=lambda m: (scores[m], m))
            logger.debug(
                "Keyword fallback alias scores: %s → best=%s",
                scores, machine_type,
            )

    # ── Symptom detection: language-level patterns only ──────────────────────
    # These map universal fault surface forms (Hindi + English) to canonical
    # symptom labels.  They are NOT machine-specific and don't need to change
    # when new machines are added to the DB.
    #
    # Rule: each keyword must be as specific as possible to its symptom bucket.
    # A bare word like "pressure" or "temperature" that could match multiple
    # symptoms must be expressed as a phrase (e.g. "pressure low", "temperature high").
    SYMPTOM_PATTERNS: Dict[str, List[str]] = {
        "not_starting": [
            "not start", "does not start", "start nahi", "no start",
            "band", "dead", "chalu nahi", "shuru nahi",
        ],
        "jammed": [
            "stuck", "jam", "jammed", "phas gaya", "phasa hua", "phas",
            "jaam", "locked up", "frozen", "seized", "atak", "atka",
            "not moving", "won't move", "wont move", "stuck in place",
            "hil nahi raha", "ruk gaya", "atak gaya",
        ],
        "humming": [
            "hum", "humming", "buzz", "buzzing", "gunguna",
            "awaz", "awaaz", "gharrr", "ghurr",
        ],
        "no_water": [
            "no water", "paani nahi", "pani nahi", "paani band",
            "water nahi", "discharge nahi",
        ],
        "low_pressure": [
            "low pressure", "pressure low", "pressure kam",
            "kam pressure", "paani kam", "water pressure low",
            "bohot kam pressure", "pressure nahi",
        ],
        "overheating": [
            "overheat", "overheating", "garam", "bahut garam",
            "heat", "hot", "temperature high", "garmi",
        ],
        "vibration": [
            "vibration", "vibrate", "vibrating",
            "kampan", "hila", "hilna", "shake", "shaking",
        ],
        "smoke": [
            "smoke", "dhuan", "dhuwa", "dhuwaan",
            "black smoke", "white smoke", "blue smoke",
        ],
        "sparks": [
            "spark", "sparks", "sparking", "chingari",
        ],
        "tripping": [
            "trip", "tripping", "mcb", "breaker",
            "overload", "relay trip",
        ],
        "no_power": [
            "no power", "no voltage", "current nahi", "light nahi",
            "bijli nahi", "power nahi",
        ],
        "oil_leak": [
            "oil leak", "tel tapak", "tel girna", "leaking oil",
        ],
        "noise": [
            "noise", "abnormal sound", "khat khat", "tak tak",
            "grinding noise", "knocking",
        ],
        # FIX 4 sub-fix 1: Temporal / frequency cues for intermittent faults.
        # "subah" (morning) and similar time-of-day words signal cold-start or
        # intermittent issues that require a specific retrieval variant.
        "intermittent": [
            "subah", "raat", "kabhi kabhi", "sometimes", "sometimes starts",
            "occasionally", "morning", "after long use", "thodi der baad",
        ],
    }

    # BUG 1 FIX: Use word-boundary regex instead of raw `kw in q` substring
    # match.  "hot" in "bohot" was True; "hum" in "human" would be True; etc.
    # re.search with \b ensures keywords match only at real token boundaries.
    symptoms: List[str] = []
    for label, keywords in SYMPTOM_PATTERNS.items():
        if any(re.search(rf'\b{re.escape(kw)}\b', q) for kw in keywords):
            symptoms.append(label)

    symptoms = list(dict.fromkeys(symptoms))

    # ── Extract component names from query using machine registry ───────────
    components = _extract_components_from_query(query, machine_type)
    if components:
        logger.debug("Keyword fallback components: %s", components)

    # ── Build enriched variant ────────────────────────────────────────────────
    variants = [query]
    if machine_type != "unknown":
        enriched_parts = [machine_type.replace("_", " ")]
        enriched_parts.extend(symptoms)
        enriched_parts.extend(components)
        enriched = " ".join(enriched_parts).strip()
        if enriched != query:
            variants.append(enriched)

    # Supplement from dynamic expansion bank
    bank_variants = _lookup_expansion_bank(machine_type, symptoms)
    all_variants = list(dict.fromkeys(variants + bank_variants))[:5]

    # Hinglish detection: any Hindi surface form in the query
    _HINDI_MARKERS = {"paani", "nahi", "hai", "garam", "awaz", "band", "chalu"}
    is_hinglish = any(w in q for w in _HINDI_MARKERS)

    logger.info(
        "Keyword fallback: machine=%s symptoms=%s components=%s variants=%d",
        machine_type, symptoms, components, len(all_variants),
    )

    return RouterOutput(
        machine_type=machine_type,
        symptoms=symptoms,
        confidence=0.50,
        language="mixed" if is_hinglish else "en",
        raw_query=query,
        query_variants=all_variants[:5],
        router_ok=False,
        error="Keyword fallback (LLM unavailable)",
    )


# ── Public API ─────────────────────────────────────────────────────────────────

async def route_query(query: str) -> RouterOutput:
    """
    Main entry point. Call this once per user query before hitting ChromaDB.

    Returns a RouterOutput with machine_type + symptoms + query_variants.
    Never raises — on LLM failure returns router_ok=False with keyword fallback.

    Requires load_machine_registry(vector_db) to have been called at startup.

    Example:
        load_machine_registry(vector_db)          # once at startup
        result = await route_query("pump start nahi")
        context, score, n = retrieve_with_confidence(
            vector_db, build_enriched_query(result),
            result.machine_type, query_variants=result.query_variants,
        )
    """
    if not query or not query.strip():
        return RouterOutput(
            machine_type="unknown", symptoms=[], confidence=0.0,
            language="en", raw_query=query, query_variants=[],
            router_ok=False, error="Empty query",
        )

    prompt = _build_router_prompt(query.strip())

    try:
        raw_response = await asyncio.to_thread(
            lambda: groq_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )
        )
        return _parse_router_json(raw_response.choices[0].message.content, query)  # MIGRATED: Gemini → Groq

    except Exception as exc:
        logger.error("Router LLM call failed: %s", exc)
        fallback = _keyword_fallback(query)
        logger.info(
            "Router keyword fallback: machine=%s symptoms=%s",
            fallback.machine_type, fallback.symptoms,
        )
        return fallback


def build_enriched_query(router_output: RouterOutput) -> str:
    parts = []
    if router_output.machine_type != "unknown":
        parts.append(router_output.machine_type.replace("_", " "))
    if router_output.symptoms:
        parts.extend(router_output.symptoms)
    raw = (router_output.raw_query or "").strip()
    if raw:
        clean = re.sub(r"[`'']", "", raw)
        parts.append(clean)
    return " ".join(parts)