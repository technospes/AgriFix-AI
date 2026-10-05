"""
multihop_diagnosis.py — AgriFix Multi-Hop Diagnostic Chain v2.0
================================================================
CHANGES v2.0:
  FIX 4: Graph-assisted query expansion — uses component relationship graph
          to automatically expand partial symptoms into related components.
          Examples:
            "pump no water"   → expands with: impeller shaft seal suction valve
            "breaker trips"   → expands with: winding insulation starter overload relay capacitor
          Runs BEFORE subsystem classification to improve ChromaDB recall.

FIX 1 (audit): Replaced _clean_json() + json.loads() with repair_json() from
               utils.helpers. repair_json() handles unterminated strings,
               single-quoted keys, and trailing commas before ] as well as }.
               Also fixed asyncio.get_event_loop().run_in_executor() →
               asyncio.to_thread() at both Gemini call sites (was already fixed
               elsewhere in the codebase per CURRENT STATE item 9).

Design:
  • Lightweight in-memory component graph (no external graph DB)
  • Traverses 1 hop from each symptom-mapped component
  • Deduplicates expanded terms before adding to enriched query
  • Chain still skips on high-confidence single-hop path

HOW TO INTEGRATE:
  In diagnosis_service.py:
    from multihop_diagnosis import run_diagnostic_chain, DiagnosticChain
    chain = await run_diagnostic_chain(machine_type, symptoms, rag_context,
                                       router_confidence, n_rag_chunks)
    enriched_query = chain.enriched_query
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# MIGRATED: Gemini → Groq — google.generativeai removed
from utils.groq_client import SHORT_CONFIG, groq_chat_completion

# FIX 1: Import repair_json instead of using local _clean_json + json.loads
from utils.json_repair import repair_json

logger = logging.getLogger(__name__)

_COMPONENT_GRAPH: Dict[str, List[str]] = {
    # Water pump / electric motor failures
    "capacitor":        ["motor winding", "starting torque", "run capacitor", "start relay"],
    "motor winding":    ["insulation", "stator", "rotor", "overload relay", "capacitor"],
    "impeller":         ["shaft", "mechanical seal", "wear ring", "suction valve", "discharge valve"],
    "mechanical seal":  ["shaft", "seal face", "water leak", "bearing"],
    "bearing":          ["shaft", "vibration", "noise", "lubrication", "grease"],
    "suction":          ["foot valve", "suction pipe", "air lock", "strainer", "prime"],
    "foot valve":       ["suction pipe", "strainer", "air lock", "check valve"],
    "prime":            ["foot valve", "air lock", "suction pipe", "priming plug"],
    "no water":         ["impeller", "foot valve", "air lock", "suction", "discharge"],
    "no discharge":     ["impeller", "discharge valve", "air lock", "suction", "blockage"],
    "humming":          ["capacitor", "bearing", "single phase", "impeller seized"],
    "not starting":     ["capacitor", "motor winding", "overload relay", "power supply", "mcb"],
    "overload":         ["motor winding", "impeller blocked", "bearing", "overload relay"],
    "mcb":              ["winding", "overload relay", "capacitor", "short circuit", "earth fault"],
    "tripped":          ["winding", "insulation", "starter", "overload relay", "capacitor"],
    "breaker":          ["winding", "insulation", "starter", "overload relay", "capacitor", "short circuit"],
    "vibration":        ["bearing", "shaft", "impeller", "coupling", "foundation bolt"],
    "noise":            ["bearing", "impeller", "cavitation", "air lock", "loose parts"],
    "overheat":         ["cooling", "winding insulation", "overload", "lubrication", "bearing"],
    "smoke":            ["winding", "insulation failure", "short circuit", "overload"],
    "leaking":          ["mechanical seal", "shaft seal", "gland packing", "pipe joint", "gasket"],

    # Tractor / diesel engine
    "fuel":             ["injector", "fuel filter", "fuel pump", "fuel line", "air in fuel"],
    "injector":         ["nozzle", "injection pump", "fuel filter", "air bleed"],
    "air filter":       ["fuel mixture", "carburetor", "throttle", "intake"],
    "cooling":          ["radiator", "thermostat", "water pump", "coolant level", "fan belt"],
    "radiator":         ["coolant", "thermostat", "fan belt", "head gasket"],
    "battery":          ["alternator", "starter motor", "wiring", "terminal", "voltage"],
    "starter":          ["battery", "solenoid", "ring gear", "ignition switch"],
    "hydraulic":        ["hydraulic pump", "control valve", "relief valve", "cylinder seal", "oil level"],
    "pto":              ["pto shaft", "pto clutch", "gearbox", "shear bolt"],
    "belt":             ["pulley", "tensioner", "belt tension", "bearing", "alignment"],

    # Generator
    "avr":              ["alternator winding", "voltage regulator", "capacitor", "output voltage"],
    "no voltage":       ["avr", "capacitor", "alternator winding", "circuit breaker", "excitation"],
    "alternator":       ["avr", "winding", "capacitor", "brush", "slip ring"],
}

# ── Symptom → component mapping (entry points into the graph) ─────────────────

_SYMPTOM_TO_COMPONENT: Dict[str, List[str]] = {
    "not starting":          ["capacitor", "not starting", "mcb"],
    "start nahi":            ["capacitor", "not starting", "mcb"],
    "humming":               ["humming", "capacitor", "bearing"],
    "buzzing":               ["humming", "capacitor"],
    "no water":              ["no water", "impeller", "foot valve"],
    "paani nahi":            ["no water", "impeller", "foot valve"],
    "no discharge":          ["no discharge", "impeller", "air lock"],
    "low discharge":         ["impeller", "suction", "air lock"],
    "vibration":             ["vibration", "bearing", "impeller"],
    "noise":                 ["noise", "bearing", "impeller"],
    "tripping":              ["tripped", "breaker", "overload"],
    "breaker trips":         ["breaker", "tripped", "winding"],
    "motor trips":           ["breaker", "tripped", "overload"],
    "overheat":              ["overheat", "bearing", "cooling"],
    "overheating":           ["overheat", "cooling", "radiator"],
    "smoke":                 ["smoke", "winding"],
    "leak":                  ["leaking", "mechanical seal"],
    "leaking":               ["leaking", "mechanical seal"],
    "no power":              ["no voltage", "mcb", "capacitor"],
    "no voltage":            ["no voltage", "avr", "alternator"],
    "fuel":                  ["fuel", "injector"],
    "diesel":                ["fuel", "injector", "air filter"],
    "oil":                   ["lubrication", "overheat", "bearing"],
}


def _graph_expand(symptoms: List[str], machine_type: str, max_terms: int = 8) -> List[str]:
    """
    FIX 4: Expand symptoms using the component relationship graph.

    Walk one hop from each symptom-mapped component and collect
    related component terms. Returns a deduplicated list of expansion
    terms (excluding the original symptom words themselves).

    Args:
        symptoms    List of English symptom phrases from RouterOutput
        machine_type  For machine-specific filtering (future use)
        max_terms   Maximum expansion terms to return

    Returns:
        List of component/keyword expansion terms ready to append to enriched query.

    Example:
        symptoms = ["no water flow", "pump running"]
        → entry nodes: ["no water", "impeller", "foot valve"]
        → 1-hop neighbors: ["impeller", "foot valve", "air lock", "suction", ...]
        → returns: ["impeller", "foot valve", "air lock", "suction", "discharge"]
    """
    symptom_text = " ".join(s.lower() for s in symptoms)
    entry_nodes: List[str] = []

    # Find entry points from symptom text
    for symptom_kw, components in _SYMPTOM_TO_COMPONENT.items():
        if symptom_kw in symptom_text:
            entry_nodes.extend(components)

    if not entry_nodes:
        logger.debug("Graph expansion: no entry nodes for symptoms=%s", symptoms)
        return []

    # 1-hop traversal
    expanded: List[str] = []
    visited: Set[str] = set(entry_nodes)

    for node in entry_nodes:
        neighbors = _COMPONENT_GRAPH.get(node, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                expanded.append(neighbor)

    # Deduplicate and cap
    result = list(dict.fromkeys(expanded))[:max_terms]
    logger.info("Graph expansion: symptoms=%s → %d terms: %s", symptoms, len(result), result)
    return result


# ── Subsystem map per machine ──────────────────────────────────────────────────

_SUBSYSTEM_MAP = {
    "water_pump":        ["motor", "impeller", "capacitor", "suction_system",
                          "discharge_system", "mechanical_seal", "bearings", "power_supply"],
    "submersible_pump":  ["motor_winding", "capacitor", "pump_stage", "foot_valve",
                          "rising_main", "control_panel", "cable"],
    "electric_motor":    ["stator_winding", "rotor", "capacitor", "bearings",
                          "terminal_box", "shaft_coupling", "power_supply"],
    "tractor":           ["engine", "hydraulic_system", "electrical", "transmission",
                          "cooling_system", "fuel_system", "pto", "steering"],
    "diesel_engine":     ["fuel_system", "air_system", "cooling_system", "lubrication",
                          "starting_system", "governor", "injectors"],
    "harvester":         ["threshing_drum", "cutting_platform", "feed_system",
                          "cleaning_system", "grain_tank", "engine", "hydraulics"],
    "generator":         ["engine", "alternator_winding", "avr", "capacitor",
                          "fuel_system", "circuit_breaker", "cooling"],
    "default":           ["mechanical", "electrical", "hydraulic", "thermal",
                          "lubrication", "fuel", "structural"],
}


# ── Causal chain template ──────────────────────────────────────────────────────

_CAUSAL_CHAIN_TEMPLATE = """\
You are a farm machinery FMEA (Failure Mode and Effects Analysis) expert.
Given the machine type, the faulty subsystem, and the reported symptoms,
return a ranked list of probable root causes with probability weights.

Machine: {machine_type}
Faulty subsystem: {subsystem}
Symptoms: {symptoms}
Context from manual: {context_snippet}

Rules:
1. Return ONLY JSON — no markdown, no preamble.
2. List 2-4 root causes, ranked by probability (highest first).
3. Each cause must include a specific inspection step (one sentence).
4. Probabilities must sum to 1.0.
5. Use plain English — no jargon.

Return EXACTLY:
{{
  "root_causes": [
    {{
      "rank": 1,
      "cause": "<specific root cause>",
      "probability": 0.45,
      "inspect": "<one concrete inspection action>"
    }}
  ],
  "recommended_first_check": "<the single most important first action>"
}}
"""

_SUBSYSTEM_TEMPLATE = """\
You are a farm machinery fault localiser.
Given the machine and symptoms, identify the faulty subsystem.

Machine: {machine_type}
Symptoms: {symptoms}
Available subsystems: {subsystems}

Return ONLY JSON:
{{
  "subsystem": "<one subsystem from the list above>",
  "confidence": 0.0,
  "reasoning": "<one sentence>"
}}
"""


# ── Output dataclasses ─────────────────────────────────────────────────────────

@dataclass
class RootCause:
    rank: int
    cause: str
    probability: float
    inspect: str


@dataclass
class DiagnosticChain:
    """Full multi-hop diagnostic result. All fields safe to access — never None."""
    machine_type:    str
    symptoms:        List[str]
    subsystem:       str              = "unknown"
    subsystem_conf:  float            = 0.0
    root_causes:     List[RootCause]  = field(default_factory=list)
    recommended_first_check: str      = ""
    enriched_query:  str              = ""
    graph_terms:     List[str]        = field(default_factory=list)  # FIX 4: graph expansion terms
    chain_ok:        bool             = True
    error:           Optional[str]    = None

    def top_cause(self) -> str:
        return self.root_causes[0].cause if self.root_causes else ""

    def as_log_dict(self) -> dict:
        return {
            "subsystem":      self.subsystem,
            "subsystem_conf": round(self.subsystem_conf, 2),
            "top_cause":      self.top_cause(),
            "n_causes":       len(self.root_causes),
            "first_check":    self.recommended_first_check[:80],
            "graph_terms":    self.graph_terms,
            "chain_ok":       self.chain_ok,
        }

def _llm_call(prompt: str) -> str:
    """Call Groq LLM with automatic primary→fallback failover (P1-2 fix)."""
    resp = groq_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        **SHORT_CONFIG,
    )
    return resp.choices[0].message.content


# ── Stage 2: Subsystem classification ─────────────────────────────────────────

async def _classify_subsystem(
    machine_type: str,
    symptoms: List[str],
) -> Tuple[str, float]:
    subsystems = _SUBSYSTEM_MAP.get(machine_type, _SUBSYSTEM_MAP["default"])
    symptom_str = "; ".join(symptoms) if symptoms else "unspecified fault"

    prompt = _SUBSYSTEM_TEMPLATE.format(
        machine_type=machine_type.replace("_", " "),
        symptoms=symptom_str,
        subsystems=", ".join(subsystems),
    )

    try:
        # FIX 1: asyncio.to_thread() replaces deprecated get_event_loop().run_in_executor()
        raw = await asyncio.to_thread(_llm_call, prompt)
        # FIX 1: repair_json() replaces json.loads(_clean_json(raw))
        data = repair_json(raw)
        subsystem = str(data.get("subsystem", "unknown")).lower().strip()
        if subsystem not in subsystems:
            matched = next((s for s in subsystems if s in subsystem or subsystem in s), "unknown")
            subsystem = matched
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        logger.info("Stage 2 subsystem: %s (conf=%.2f)", subsystem, confidence)
        return subsystem, confidence
    except Exception as exc:
        logger.warning("Stage 2 subsystem classification failed: %s", exc)
        return "unknown", 0.0


# ── Stage 3/4: Root cause ranking ─────────────────────────────────────────────

async def _rank_root_causes(
    machine_type: str,
    subsystem: str,
    symptoms: List[str],
    context_snippet: str,
) -> Tuple[List[RootCause], str]:
    symptom_str = "; ".join(symptoms) if symptoms else "unspecified"
    ctx_snippet = context_snippet[:500] if context_snippet else "no manual context available"

    prompt = _CAUSAL_CHAIN_TEMPLATE.format(
        machine_type=machine_type.replace("_", " "),
        subsystem=subsystem.replace("_", " "),
        symptoms=symptom_str,
        context_snippet=ctx_snippet,
    )

    try:
        # FIX 1: asyncio.to_thread() replaces deprecated get_event_loop().run_in_executor()
        raw = await asyncio.to_thread(_llm_call, prompt)
        # FIX 1: repair_json() replaces json.loads(_clean_json(raw))
        data = repair_json(raw)

        causes = []
        for item in data.get("root_causes", [])[:4]:
            causes.append(RootCause(
                rank        = int(item.get("rank", 99)),
                cause       = str(item.get("cause", "")).strip(),
                probability = max(0.0, min(1.0, float(item.get("probability", 0.25)))),
                inspect     = str(item.get("inspect", "")).strip(),
            ))
        causes.sort(key=lambda c: c.rank)
        first_check = str(data.get("recommended_first_check", "")).strip()

        logger.info(
            "Stage 3/4 root causes: %s",
            [(c.cause[:40], round(c.probability, 2)) for c in causes],
        )
        return causes, first_check

    except Exception as exc:
        logger.warning("Stage 3/4 root cause ranking failed: %s", exc)
        return [], ""


# ── Build enriched query from chain output ─────────────────────────────────────

def _build_enriched_query(
    machine_type: str,
    symptoms: List[str],
    subsystem: str,
    root_causes: List[RootCause],
    graph_terms: List[str],
) -> str:
    """
    Combine all diagnostic chain signals + graph expansion terms into a single
    enriched ChromaDB query string.

    Example output:
      "electric motor capacitor not starting humming failed start capacitor
       seized bearing power supply single phase motor winding overload relay"
    """
    parts = [machine_type.replace("_", " ")]
    if subsystem and subsystem != "unknown":
        parts.append(subsystem.replace("_", " "))
    parts.extend(symptoms[:3])
    # Add top-2 cause keywords
    for cause in root_causes[:2]:
        words = cause.cause.lower().split()[:5]
        parts.extend(w for w in words if len(w) > 3)
    # FIX 4: Add graph expansion terms
    parts.extend(graph_terms[:5])

    return " ".join(dict.fromkeys(parts))  # deduplicate, preserve order


# ── Public API ─────────────────────────────────────────────────────────────────

async def run_diagnostic_chain(
    machine_type: str,
    symptoms: List[str],
    rag_context: str = "",
    router_confidence: float = 1.0,
    n_rag_chunks: int = 6,
) -> DiagnosticChain:
    """
    Run the full multi-hop diagnostic chain with graph-assisted expansion.

    Args:
        machine_type        from RouterOutput.machine_type
        symptoms            from RouterOutput.symptoms
        rag_context         initial RAG context string (for cause ranking context)
        router_confidence   from RouterOutput.confidence (skip chain if high)
        n_rag_chunks        number of RAG chunks retrieved (skip if enough)

    Returns DiagnosticChain with .enriched_query ready for a second retrieve call.
    .graph_terms contains the component graph expansion terms (for logging/debug).

    FIX 4: Graph expansion always runs (even on high-confidence fast-path)
    to enrich the query with related component terms at zero LLM cost.

    Skip conditions for LLM stages only:
      • router_confidence >= 0.80 AND n_rag_chunks >= 4
        → skip LLM classification, but still apply graph expansion
    """
    # FIX 4: Always run graph expansion (no LLM cost)
    graph_terms = _graph_expand(symptoms, machine_type)

    # Fast-path: high-confidence single-hop — skip LLM classification
    if router_confidence >= 0.80 and n_rag_chunks >= 4:
        simple_parts = [machine_type.replace("_", " ")] + symptoms[:3] + graph_terms[:4]
        simple_query = " ".join(dict.fromkeys(simple_parts))
        logger.info(
            "Diagnostic chain fast-path: graph_terms=%s enriched='%s'",
            graph_terms, simple_query[:80],
        )
        return DiagnosticChain(
            machine_type=machine_type,
            symptoms=symptoms,
            enriched_query=simple_query,
            graph_terms=graph_terms,
            chain_ok=True,
        )

    logger.info(
        "Running multi-hop diagnostic chain: machine=%s symptoms=%s graph_terms=%s",
        machine_type, symptoms, graph_terms,
    )

    try:
        # Stage 2: subsystem classification
        subsystem, subsystem_conf = await _classify_subsystem(machine_type, symptoms)

        # Stage 3/4: root cause ranking
        root_causes, first_check = await _rank_root_causes(
            machine_type, subsystem, symptoms, rag_context
        )

        # Build enriched query including graph expansion terms
        enriched = _build_enriched_query(
            machine_type, symptoms, subsystem, root_causes, graph_terms
        )

        chain = DiagnosticChain(
            machine_type=machine_type,
            symptoms=symptoms,
            subsystem=subsystem,
            subsystem_conf=subsystem_conf,
            root_causes=root_causes,
            recommended_first_check=first_check,
            enriched_query=enriched,
            graph_terms=graph_terms,
            chain_ok=True,
        )
        logger.info("Diagnostic chain complete: %s", chain.as_log_dict())
        return chain

    except Exception as exc:
        logger.error("Diagnostic chain failed: %s", exc)
        # Even on failure, include graph terms in fallback query
        fallback_parts = [machine_type.replace("_", " ")] + symptoms[:3] + graph_terms[:3]
        fallback_query = " ".join(dict.fromkeys(fallback_parts))
        return DiagnosticChain(
            machine_type=machine_type,
            symptoms=symptoms,
            enriched_query=fallback_query,
            graph_terms=graph_terms,
            chain_ok=False,
            error=str(exc),
        )


# ── Integration example for diagnosis_service.py ──────────────────────────────
"""
In diagnosis_service.py, inside generate_diagnosis_with_gemini():

    from multihop_diagnosis import run_diagnostic_chain

    chain = await run_diagnostic_chain(
        machine_type=machine_type,
        symptoms=router_symptoms or [],
        rag_context=effective_rag,
        router_confidence=router_confidence,
        n_rag_chunks=n_chunks_estimate,
    )

    if chain.enriched_query and vector_db is not None:
        rag_v2, score_v2, n_v2 = retrieve_with_confidence(
            vector_db, chain.enriched_query, machine_type
        )
        if score_v2 > top_score and rag_v2:
            effective_rag, top_score = rag_v2, score_v2

    # chain.graph_terms available for logging/debug
    # chain.recommended_first_check can seed the LLM prompt
"""


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    # Test graph expansion without LLM
    print("\n=== Graph Expansion Tests ===")
    tests = [
        (["not starting", "humming"], "water_pump"),
        (["no water", "pump running"], "water_pump"),
        (["breaker trips on startup"], "electric_motor"),
        (["paani nahi aa raha", "motor chal raha"], "water_pump"),
    ]
    for symptoms, machine in tests:
        terms = _graph_expand(symptoms, machine)
        print(f"\nSymptoms: {symptoms}")
        print(f"Machine:  {machine}")
        print(f"Expanded: {terms}")

    if not os.environ.get("GROQ_API_KEY"):  # MIGRATED: Gemini → Groq
        print("\nSet GROQ_API_KEY for LLM chain test.")  # MIGRATED: Gemini → Groq
    else:
        from dotenv import load_dotenv
        load_dotenv()

        async def _test():
            result = await run_diagnostic_chain(
                machine_type="water_pump",
                symptoms=["not starting", "humming noise", "no water flow"],
                rag_context="",
                router_confidence=0.5,
                n_rag_chunks=1,
            )
            print("\nDiagnostic Chain Result:")
            print(f"  Subsystem:   {result.subsystem} ({result.subsystem_conf:.0%})")
            print(f"  Top cause:   {result.top_cause()}")
            print(f"  First check: {result.recommended_first_check}")
            print(f"  Graph terms: {result.graph_terms}")
            print(f"  Query:       {result.enriched_query}")

        asyncio.run(_test())