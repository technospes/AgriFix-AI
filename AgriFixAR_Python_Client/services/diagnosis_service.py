from __future__ import annotations
import asyncio
import hashlib
import io
import json
import logging
import re
from typing import List, Optional, TypedDict
from enum import StrEnum
import os
# MIGRATED: Gemini → Groq — google.generativeai removed
from utils.groq_client import groq_client, TEXT_MODEL, JSON_CONFIG, groq_chat_completion  # MIGRATED: Gemini → Groq  # FAILOVER: primary → fallback
from utils.json_repair import repair_json
from PIL import Image

from rag import retrieve_with_confidence, RAG_WEAK_THRESHOLD
from query_router import route_query, load_machine_registry
from utils.helpers import generate_cache_key, get_cached_response, cache_response
from utils.machine_registry import (
    get_profile_or_default, get_allowed_area_ids, get_compact_parts_list,
    get_compact_safety_keywords, is_electric_machine, is_tractor_attachment,
)
from services.safety_guards import (
    run_text_hazard_checks,
    _guard_electric_hazard,
    _guard_dangerous_workaround,
    _guard_emergency_hazard,
)
# FIX 3: Multi-hop diagnostic chain — subsystem classification + root cause ranking
from multihop_diagnosis import run_diagnostic_chain

# FIX 5: Procedure validator — safety step injection before response is returned
from procedure_validator import validate_procedure

# FIX 6: Adaptive clarification loop — ask targeted questions before escalating
# clarification_loop imports removed — orchestrator handles clarification exclusively

# FIX 6: Verification gate — verify machine/component before final repair
from services.verification_service import verify_step_with_gemini

# Repair-plan structural validation — single source of truth, shared with
# repair_agent.py. See agent/validation.py for the rationale.
from agent.validation import InvalidRepairPlan, validate_repair_plan_steps

# Deterministic jargon backstop — shared with repair_agent.py. See
# agent/jargon_guard.py for the rationale (same pattern as the tool /
# interaction validation already used in this pipeline).
from agent.jargon_guard import apply_jargon_guard


logger = logging.getLogger(__name__)
# _GEMINI_MODEL removed — TEXT_MODEL from groq_client used instead  # MIGRATED: Gemini → Groq
_RAG_STRICT_THRESHOLD = 0.40

# ── Escalation response schema ────────────────────────────────────────────────

class RagSource(StrEnum):
    """Canonical rag_source values. One typo breaks analytics — use these."""
    PRE_CALL            = "pre_call_guard"
    ERROR               = "error"
    NO_CONTEXT          = "no_context"
    EMPTY_RAG_MAINT     = "guard5_empty_rag_maintenance"
    UNKNOWN_ATTACHMENT  = "guard_unknown_attachment"
    ELECTRIC_HAZARD     = "pre_guard_electric_shock_injury"
    WORKAROUND          = "pre_guard_workaround"
    EMERGENCY_HAZARD    = "emergency_hazard_guard"
    ESCALATION          = "escalation"


class EscalationResponse(TypedDict, total=False):
    """Canonical shape of every escalation response. Flutter parses this."""
    status: str
    problem_description: str
    technical_analysis: str
    solution: dict
    rag_source: str
    machine_label: str
    # Optional: unsafe scene
    unsafe_scene_suspected: bool
    unsafe_scene_message: str
    # Optional: emergency hazard metadata
    hazard_category: str
    hazard_level: str
    emergency_stop_required: bool
    requires_power_isolation: bool
    requires_medical_attention: bool
    requires_fire_response: bool
    requires_evacuation: bool
    # Internal markers
    cache_hit: bool
    _informational_override: bool
    _reassurance_override: bool
    _verification_issues: list

# ── FIX 6: Verification gate thresholds ──────────────────────────────────────
# Below this confidence, verification fails and we return "Need verification"
_VERIFICATION_CONFIDENCE_THRESHOLD = 0.60
# Only run the verification gate when risk_level is HIGH or CRITICAL
_VERIFICATION_RISK_LEVELS = {"HIGH", "CRITICAL"}

_GROUNDING_RULE = """\
GROUNDING (non-negotiable): retrieved chunks below are your ONLY diagnosis source. Which part is at fault, the fix, and any spec/interval/measurement MUST come exclusively from KNOWLEDGE BASE CONTEXT. Never invent a cause, fix, spec, or interval.
General mechanical knowledge may ONLY: connect evidence across chunks, phrase things farmer-friendly, describe what a named part looks like/where it sits, or fill trivial procedural gaps.
Never contradict the retrieved evidence. If evidence can't diagnose the fault: "This is not covered in the manual — consult a certified mechanic." """

_WEAK_CONTEXT_HEADER = """\
RAG QUALITY: LOW (top score < 0.40). STRICT MODE:
- State only what the chunks explicitly say.
- No speculation, no generalising, no outside-knowledge gap-filling.
- Fault/procedure not covered → say so and escalate.
"""

_STRONG_CONTEXT_HEADER = """\
RAG QUALITY: STRONG (top score ≥ 0.40). Manual excerpts are highly relevant — use as primary source.
"""

_NO_CONTEXT_ESCALATION_EN = "I was unable to find relevant information in the technical manual for this specific situation. Please consult a certified mechanic or your nearest Mahindra service centre for a safe diagnosis."
_NO_CONTEXT_ESCALATION_HI = "इस समस्या के लिए हमारे तकनीकी मैनुअल में कोई जानकारी नहीं मिली। कृपया प्रमाणित मैकेनिक या नजदीकी महिंद्रा सर्विस सेंटर से सुरक्षित निदान करवाएं।"

def _has_visual_context_text(problem_text: str) -> bool:
    return "visual context:" in (problem_text or "").lower()

def _extract_visual_snippet(problem_text: str) -> str:
    match = re.search(r'visual context:\s*(.+?)(?:audio:|$)', problem_text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

_SPEC_FIDELITY_RULE = """
## RULE 7 — FLUID SUBSTITUTION & SPEC FIDELITY
Non-OEM fluid query (cooking/mustard oil, engine oil as hydraulic fluid, petrol in diesel, etc.):
1. Reject the substitute explicitly + state the exact damage it causes.
2. Copy the OEM spec (grade/viscosity/standard) VERBATIM from the chunk — never paraphrase or approximate it.
3. Put that verbatim spec in BOTH technical_analysis AND a dedicated step.

Water pump suction failure / loss of prime / re-priming query:
1. Foot valve / suction pipe check is MANDATORY as a named step, even if the farmer only asked about re-priming.
2. Use the exact term "foot valve" as written in the manual.
"""

def _build_strict_grounding_prompt(
    machine_type, machine_label, problem_text, rag_context, allowed_areas,
    parts_list, safety_keywords, language, has_visual_frames, context_quality,
    top_score, router_symptoms: Optional[List[str]] = None,
) -> str:
    visual_note = ""
    visual_text_present = _has_visual_context_text(problem_text)
    visual_snippet = _extract_visual_snippet(problem_text) if visual_text_present else ""
 
    if has_visual_frames:
        visual_note = (
            "Camera images available — use to confirm machine type/condition, "
            "but base repair steps on the manual excerpts + symptom text below."
        )
    elif visual_text_present:
        visual_note = (
            f'VISUAL PRIORITY OVERRIDE ACTIVE: visual context ("{visual_snippet}") '
            f'is the PRIMARY symptom. If \'Audio:\' contradicts it, trust the visual.\n'
        )
 
    electric_note = ""
 
    quality_banner = _WEAK_CONTEXT_HEADER if context_quality == "weak" else _STRONG_CONTEXT_HEADER
    quality_banner += f"Top chunk relevance: {top_score:.2f} | "
 
    if rag_context and rag_context.strip():
        rag_block = (
            f"## MANUAL EXTRACTS (AUTHORITATIVE — USE FIRST)\n{quality_banner}\n"
            f"{rag_context}"
        )
    else:
        rag_block = f"""\
## ⚠️ NO MANUAL EXTRACTS — HALLUCINATION TRAP ACTIVE
No chunks cleared the relevance threshold. Output an ESCALATION response. \
steps = [] (empty array). Do NOT generate steps, intervals, oil grades, or \
any procedure from training knowledge — that is a CRITICAL FAILURE no \
matter how helpful it would seem.
 
- CASE A (machine matches {machine_type}, query is a real mechanical/\
maintenance task) → status="escalate", steps=[]
  technical_analysis = "This procedure is not covered in the {machine_label} service manual."
  safety_warnings_en[0] = "This specific procedure is not in our repair manual for the {machine_label}. Please consult the manufacturer's user guide or a certified mechanic."
 
- CASE B (out-of-scope / unknown fault) → status="escalate", steps=[]
  technical_analysis = "Insufficient knowledge base coverage for this symptom."
  safety_warnings_en[0] = "Automatic diagnosis unavailable: symptom outside knowledge base. Consult a certified mechanic." """
 
    target_symptoms = ", ".join(router_symptoms) if router_symptoms else problem_text
    user_query_words = problem_text
 
    grounding_rules = f"""
{_GROUNDING_RULE}
 
## STRICT GROUNDING — ZERO-HALLUCINATION PROTOCOL
 
🔵 MANDATORY CHAIN OF THOUGHT — complete all 6 steps in "internal_reasoning" \
before any other output field.
 
STEP 1 — Visual/Audio check (1 sentence): "The user describes [X symptom] \
with [visual/audio evidence Y]." Visual context (if present) always \
overrides audio/text.
 
STEP 2 — RAG chunk search & ranking. User said: "{user_query_words}" \
| extracted symptoms: {target_symptoms}
- A chunk matches if its PROBLEM field names the SAME symptom OR the SAME \
component/system, even with different wording ("clutch is stuck" matches \
"Clutch fails to work" / "Clutch cable damage" — all clutch faults; don't \
require literal word-for-word equality).
- List every matching chunk ID + its PROBLEM field + why it matches. \
⛔ Quoting the chunk content is mandatory — "a safety chunk applies" alone \
is a GROUNDING FAILURE.
- Rank matches by symptom overlap (prefer chunks covering MULTIPLE user \
symptoms). Highest-overlap chunk = PRIMARY DIAGNOSIS SOURCE. If none \
overlap at all, state "no chunk covers this symptom".
 
STEP 3 — Action decision. State exactly one of:
- "ESCALATE — Universal safety hazard (Chunk [ID]) overrides all other rules."
- "ESCALATE — Dangerous workaround detected (Chunk [ID])."
- "ESCALATE — ZERO chunks share the same symptom OR same component/system \
as the problem (RULE 5 zero-overlap test)."
- "ESCALATE — Routine maintenance, no active fault (Chunk [ID])."
- "SUCCESS/REASSURE — Chunk [ID] explicitly confirms symptom is normal."
- "SUCCESS/FIRE_EXT — informational query, ESCALATE_IF says DO NOT escalate."
- "DIAGNOSE — Chunk [ID] covers the same component/system (RULE 5 \
component-class match); use its PROBLEM/STEPS as closest applicable fix."
 
STEP 4 — Faithfulness check. For every DIAGNOSTIC claim (faulty part, fix, \
spec/interval/part number) — is it explicit in the matched chunk? If not, \
delete or soften it. Does NOT apply to general descriptive language \
(landmarks, appearance, healthy-vs-faulty) — that's required by the \
FARMER INSTRUCTION STANDARD below even when the manual is silent on it.
 
STEP 5 — technical_analysis language check (applies even on escalate, \
where solution.steps is empty and the FARMER INSTRUCTION STANDARD below \
never runs on any text). technical_analysis and safety_warnings_en are \
still farmer-facing. Before finalizing: no jargon without a plain-English \
gloss ("corrosion" alone is banned — "rust/greenish buildup" required), \
short plain sentences, no hedging chains ("could be indicative of a \
variety of problems including but not limited to..."). State plainly \
what was checked and why it's inconclusive, in one or two short \
sentences a first-time operator can read aloud.

STEP 6 — jargon_check. For every step you are about to output, look at \
the FIRST SENTENCE of "action" and of "description" only. List any \
technical part name a first-time farmer wouldn't know (e.g. shaft, \
coupling, bearing, terminal board, capacitor, gland, seal, impeller) \
that appears there before that part has been visually introduced. If \
any are found, state the rewrite you will use instead (appearance/\
shape/size/landmark, no technical name). If none are found, state \
"clean".
 
## RULE 1 — UNIVERSAL SAFETY MASTER (absolute highest priority)
STEP 2 finds a Universal_Safety_Master chunk whose ESCALATE_IF is triggered:
→ status="escalate" | safety_warnings_en[0] = chunk's STEPS field, \
CHARACTER-FOR-CHARACTER, zero paraphrasing | technical_analysis = chunk's \
PROBLEM field verbatim.
 
## RULE 2 — REASSURANCE (normal operation/readings)
Only if a chunk EXPLICITLY calls this symptom/reading normal AND no \
Universal Safety ESCALATE_IF fired: status="success", 1-2 steps explaining \
why it's normal per the chunk.
 
## RULE 3 — FIRE EXTINGUISHER QUERY (informational, no active fire)
Matched chunk's ESCALATE_IF says "DO NOT escalate": status="success", copy \
that chunk's STEPS verbatim.
 
## RULE 4 — ROUTINE MAINTENANCE REDIRECT
Maintenance-schedule query, no active fault: status="escalate", but \
solution.steps MUST be populated from the chunk's maintenance procedure — \
never leave steps empty here.
 
## RULE 5 — HALLUCINATION TRAP (strict failsafe)
Binary test, no middle ground: does ANY chunk share the SAME component/\
system as the problem (per STEP 2's matching rule) — regardless of \
whether its exact symptom words differ?
- NO chunk shares that component/system → status="escalate", steps=[]. \
Never generate steps/estimates/procedures from training knowledge.
- YES, at least one chunk shares that component/system → you MUST \
DIAGNOSE using the highest-overlap chunk as sole source of truth. A \
different symptom WORD ("jammed" vs. "vibration/noise") is NOT grounds \
to escalate when the component/system matches — escalating here is \
itself a rule violation, not caution.
WORKED EXAMPLE: problem="Motor is jammed", chunk PROBLEM="Shaft → \
Coupling → Motor — vibration, noise, seal leak". Symptom words differ, \
but component/system (motor) is the same → DIAGNOSE using that chunk. \
Escalating this case is WRONG.
 
## RULE 6 — COPY-PASTE MANDATE (universal, no exceptions)
Any rule requiring a copied STEPS field: locate the exact string in Manual \
Extracts, copy it character-for-character into the target output field.
{_SPEC_FIDELITY_RULE}
## RULE 8 — MISSING/REMOVED PART (visual override)
Visual context states a part is MISSING/REMOVED/ABSENT → treat as \
confirmed fault, proceed to DIAGNOSE, warn about consequences verbatim in \
technical_analysis.
 
## RULE 9 — FALSE POSITIVE GUARD
"Spark plug" cleaning/inspection mentioned → do NOT escalate for \
electrical hazard; proceed to DIAGNOSE.
"""
 
    output_format = f"""
## OUTPUT FORMAT — JSON ONLY, NO MARKDOWN
 
CRITICAL SCHEMA RULES:
1. "internal_reasoning" is the FIRST key, contains all 6 CoT steps.
2. "solution" always present.
3. status="escalate" → solution.steps = [] EXCEPT Rule 4 (Maintenance), \
which must have steps.
4. status="success" → solution.steps has ≥1 step object.
5. Every step has "tracking_scope": "component" (specific, locateable \
part) or "assembly" (general area, fluid check, whole machine).
6. "required_part": snake_case ID from {parts_list} if tracking_scope is \
"component"; null if "assembly".
7. "area_hint" must be one of: {allowed_areas}
 
## 🌾 FARMER INSTRUCTION STANDARD
 
Teach, don't just list steps. The farmer should feel more confident after \
each step, never overwhelmed.
 
Assume the operator: has never repaired this machine, cannot name \
components, knows no mechanical terms, may be anxious about mistakes. \
Write for confidence without sacrificing accuracy.
 
- "action": short imperative in PLAIN ENGLISH that a first-time farmer \
immediately understands ("Switch OFF the pump", "Inspect the metal \
connector"). FORBIDDEN: technical jargon, chunk headers, internal \
hierarchy labels, or arrows ("→", "->"). Never copy a manual heading. \
The "action" text MUST NOT contain the technical name of the part \
being repaired unless it was visually introduced in a previous step. \
You MUST translate the action target into a visual reference (e.g., \
write "Inspect the metal connector", NEVER "Inspect the coupling").
- "description": the teaching. Include, where relevant:
  • locating the part per COMPONENT INTRODUCTION below
  • how to visually identify it; what healthy vs. faulty looks like
  • how to do the action safely
  • how to confirm the outcome by sight/sound/touch, observations over \
measurements ("The humming noise should stop.")
  Include only what helps the CURRENT step — don't force every category \
into every step.
- "step_type": one of safety | inspection | repair | verification.
  • safety — ONLY a NEW mid-repair hazard not covered by the initial \
safety gate (the app's SafetyGate already handles startup power-off).
  • inspection — camera-verifiable visual check ONLY. FORBIDDEN: \
measurement tools, exact numbers (40-45mm). REQUIRED: specific snake_case \
required_part, visible landmark, healthy-vs-faulty description. Convert \
any manual measurement to a body-part comparison (finger-width).
  • repair — replace/tighten/remove/install/adjust; end with a camera \
verification sub-step ("point camera at [part] to verify it's correctly \
installed/tightened/replaced").
  • verification — restore power/start/confirm operation; camera-verifiable \
where possible ("point camera at [part], confirm [expected outcome]").
 
COMPONENT INTRODUCTION — PROGRESSIVE DISCLOSURE (mandatory order, first \
interaction with a part only):
1. VISUAL ANCHOR — opening sentence: colour/shape/size, only if visually \
certain from manual or camera context; never invent.
2. RECOGNITION — position vs. ONE permanent landmark: power cable, \
cooling fan, fuel tank, belt, wheel, air filter, starter motor, cooling \
fins, pump housing, frame. Forbidden: "near the component", "adjacent to \
the mechanism", or any other vague positional clue.
3. FUNCTION — one clause on what it does.
4. TECHNICAL NAME — last, only once the operator can point to the part. \
After that, just use the name unless confusable with a similar part.
JARGON QUARANTINE: The opening half of every description must be purely \
visual. Do not use the technical name of the target part, or the names \
of nearby technical parts, until the farmer could realistically point to \
the correct object. Describe by color, shape, size, and permanent \
landmarks first. Only after the object has been visually identified may \
you introduce its technical name. Once introduced, you may use it normally.
FIRST SENTENCE TEST: If the first sentence contains ANY technical part \
name a first-time farmer wouldn't know (e.g., shaft, coupling, bearing, \
terminal board, capacitor, gland, seal, impeller), rewrite it. Use only \
appearance and location.
ANTI-CIRCULAR: never define a part using its own name ("the shaft \
coupling joins the shafts") — describe the connector's shape and the rod \
it sits on instead.
HALLUCINATION GUARD: if the manual doesn't support a confident landmark, \
don't invent one — describe only what's visually certain.

GOLD EXAMPLES (learn the structure, NEVER copy the text onto a real \
part — these are deliberately unrelated to any machine you'll ever \
diagnose, so you cannot reuse their wording, only their shape):
- [Unrelated: Tractor Seat]: "A wide black cushion sits directly above \
the main rear axle. It supports the driver during operation. This is \
the operator seat."
- [Unrelated: Combine Harvester Reel]: "A massive rotating cylinder \
covered in metal teeth spans the entire front width of the machine. \
It pulls the crop into the cutting bar. This is the pickup reel."
- [Unrelated: Fictional Fluid Valve]: "A red star-shaped plastic dial \
sits on top of the main blue water tank. Turning it controls the flow \
of liquid. This is the primary pressure valve."
 
SAFETY PREREQUISITES: repeat any prerequisite immediately required for a \
step, even if stated earlier ("Ensure main power is OFF" again if this \
step touches electrical parts).
 
NEVER TELL THE FARMER TO READ THE MANUAL — the farmer may be semi-\
literate. Be the manual. BANNED PHRASES (never in action/description):
"Refer to the manual" / "Read the user guide" / "Check the manufacturer's \
instructions" / "See service manual" / "Consult a certified mechanic" / \
"Bring a technician" / "Contact Mahindra service centre" / "Visit the \
dealership" / any measurement with units ("40-45mm") / any tool-based \
measurement / chunk headers / manual hierarchy labels / "→" / "->" / \
"/" used as hierarchy separators / copied section titles from the manual.
If a professional is genuinely required (internal engine work, electrical \
hazard beyond simple checks): escalate the WHOLE diagnosis \
(status="escalate", steps=[]) — never generate steps and tack on "if \
problem persists, consult a mechanic."
 
LANGUAGE: "inspect"/"check"/"verify" only when immediately followed by \
what to look for.
GOOD: "Inspect the capacitor for bulging, oil leakage, or burn marks."
BAD: "Inspect the capacitor."
Hedge ("typically", "usually") only for general engineering knowledge; \
state manual-sourced facts confidently. Don't overuse hedging.
 
QUALITY: write only enough for the current step. No previewing future \
steps, no repeating past ones. One problem per step.
 
LAST STEP MUST BE ACTIONABLE — a real verification check ("Start the \
engine and check the clutch engages smoothly"), never "if problem \
persists, consult a mechanic" or "refer to the manual." The escalate_if \
field already owns the escalation condition — don't repeat it as a step.
 
CONSISTENCY: apply this standard to EVERY step including the last. "Check \
for other issues" with no landmarks/visuals is not acceptable — if the \
manual lists several failure points ("welding points breaking off, fork \
pin bent, spring failure"), describe where to look for each and what each \
looks like damaged.
 
Goal: safe completion, not comprehensive education.
"""
 
    prompt = f"""
You are an experienced agricultural mechanic explaining repairs to a \
first-time {machine_label} operator, guided by the manufacturer's manual. \
Reduce uncertainty. Never assume the operator knows machinery/component \
names.
{visual_note}
 
## PROBLEM DESCRIPTION
Machine: {machine_label}
Symptoms: {problem_text}
 
{rag_block}
 
{grounding_rules}
 
## SAFETY WARNINGS & LANGUAGE RULES
- Include "⚠️ ESCALATE_IF:" entries verbatim in safety_warnings_en.
- No jargon. Plain English only.
 
{output_format}
""".strip()
    return prompt

# ── Unified escalation response factory ───────────────────────────────────────
# Single source of truth for every escalation response dict. All pre-LLM guards,
# fallback paths, and no-context handlers produce responses through this function.
#
# Reserved keys (status, solution, problem_description, technical_analysis,
# safety_warnings) are protected — metadata cannot overwrite them.

import copy as _copy

_BASE_ESCALATION: dict = {
    "status": "escalate",
    "solution": {
        "steps": [],
        "tools_needed": [],
    },
}

_RESERVED_KEYS: frozenset = frozenset({
    "status", "solution", "problem_description", "technical_analysis",
    "safety_warnings_en", "safety_warnings_hi", "rag_source", "machine_label",
})


def _build_escalation_response(
    machine_type: str,
    machine_label: str,
    problem_text: str,
    technical_analysis: str,
    safety_warnings_en: List[str],
    safety_warnings_hi: List[str],
    *,
    rag_source: RagSource = RagSource.ESCALATION,
    problem_identified: str = "",
    metadata: Optional[dict] = None,
) -> dict:
    """
    Produce a canonical escalation response dict.

    Args:
        machine_type:        Canonical machine ID (e.g. "electric_motor").
        machine_label:       Human-readable label (e.g. "Electric Motor").
        problem_text:        Original farmer query text.
        technical_analysis:  One-paragraph explanation of why escalated.
        safety_warnings_en:  List of English safety messages.
        safety_warnings_hi:  List of Hindi safety messages.
        rag_source:          Canonical RagSource enum value.
        problem_identified:  Short fault label (defaults to technical_analysis).
        metadata:            Optional dict of extra fields merged into response.
                             Reserved keys are silently dropped.

    Returns:
        EscalationResponse-compatible dict.
    """
    result = _copy.deepcopy(_BASE_ESCALATION)
    result.update({
        "problem_description": problem_text,
        "technical_analysis": technical_analysis,
        "solution": {
            **result["solution"],
            "machine_type": machine_type,
            "problem_identified": problem_identified or technical_analysis,
            "safety_warnings_en": list(safety_warnings_en),
            "safety_warnings_hi": list(safety_warnings_hi),
        },
        "rag_source": rag_source.value,
        "machine_label": machine_label,
    })

    # Merge caller-supplied metadata, protecting reserved keys
    if metadata:
        for key, value in metadata.items():
            if key not in _RESERVED_KEYS:
                result[key] = value

    return result

def _should_hard_escalate(problem_text: str, rag_context: str) -> Optional[str]:
    if not problem_text or not problem_text.strip(): return "__empty_input__"
    return None

_MAINTENANCE_PATTERNS = [
    re.compile(r'\boil\s+change\b', re.IGNORECASE),
    re.compile(r'\boil\s+(?:change\s+)?interval\b', re.IGNORECASE),
    re.compile(r'\bfilter\s+(?:change|replace|interval)\b', re.IGNORECASE),
    re.compile(r'\bgrease\s+(?:schedule|interval|nipple|point)\b', re.IGNORECASE),
    re.compile(r'\bservice\s+interval\b', re.IGNORECASE),
]

def _guard5_empty_rag_maintenance(problem_text: str, rag_context: str, machine_type: str, machine_label: str) -> Optional[dict]:
    if rag_context and rag_context.strip(): return None
    if not any(p.search(problem_text or "") for p in _MAINTENANCE_PATTERNS): return None
    logger.warning(f"🛡️ GUARD 5 PRE-LLM: Empty RAG + maintenance query [{machine_type}] — escalating")
    return _build_escalation_response(
        machine_type=machine_type,
        machine_label=machine_label,
        problem_text=problem_text,
        technical_analysis=f"This maintenance procedure is not covered in the {machine_label} service manual.",
        safety_warnings_en=[f"This specific procedure is not in our repair manual for the {machine_label}. Please consult the manufacturer's user guide or a certified mechanic."],
        safety_warnings_hi=[f"यह रखरखाव की जानकारी हमारे {machine_label} मैनुअल में नहीं है। कृपया निर्माता की गाइड देखें या प्रमाणित मैकेनिक से पूछें।"],
        rag_source=RagSource.EMPTY_RAG_MAINT,
        problem_identified="Routine maintenance schedule query — not in knowledge base.",
    )


# ─── PRE-LLM GUARDS ─────────────────────────────────────────────────────────

_UNKNOWN_ATTACHMENT_PATTERNS = [
    re.compile(r'\bunknown\s+attachment\b', re.IGNORECASE),
    re.compile(r'\bcannot\s+identify.*\battachment\b', re.IGNORECASE),
    re.compile(r'\blaser\s+land\s+level', re.IGNORECASE),
    re.compile(r'\bunrecognized\s+(?:attachment|implement|tool)\b', re.IGNORECASE),
]

def _guard_unknown_attachment(problem_text: str, machine_type: str, machine_label: str) -> Optional[dict]:
    if not any(p.search(problem_text or "") for p in _UNKNOWN_ATTACHMENT_PATTERNS): return None
    logger.warning(f"🛡️ GUARD UNKNOWN_ATTACHMENT: Unidentified implement [{machine_type}]")
    return _build_escalation_response(
        machine_type=machine_type,
        machine_label=machine_label,
        problem_text=problem_text,
        technical_analysis="The attachment or implement cannot be identified. This falls outside the knowledge base coverage for this machine.",
        safety_warnings_en=["This attachment is not covered in our repair manual. Consult the manufacturer."],
        safety_warnings_hi=["यह उपकरण हमारे मैनुअल में नहीं है।"],
        rag_source=RagSource.UNKNOWN_ATTACHMENT,
        problem_identified="Unknown implement — insufficient knowledge base coverage.",
    )

# ─── NON-DESTRUCTIVE POST-LLM GUARDS ────────────────────────────────────────

_FIRE_EXT_PROBLEM_KEYWORDS = ["fire extinguisher", "agni shamak", "agnisamak", "fire cylinder"]
_ACTIVE_FIRE_KEYWORDS      = ["aag lagi", "jal raha", "flames", "burning", "smoke coming", "dhuan aa raha", "active fire", "sparks"]

def _guard1_fire_extinguisher(diagnosis, problem_text, rag_context, machine_type, machine_label):
    prob_lower = (problem_text or "").lower()
    if not any(kw in prob_lower for kw in _FIRE_EXT_PROBLEM_KEYWORDS): return None
    if any(kw in prob_lower.replace("fire extinguisher", "").replace("fire cylinder", "") for kw in _ACTIVE_FIRE_KEYWORDS): return None
    logger.info("🛡️ GUARD 1 NON-DESTRUCTIVE OVERRIDE: Fire extinguisher informational")
    diagnosis["status"] = "success"
    diagnosis["_informational_override"] = True
    diagnosis["technical_analysis"] = "User asks how to use a mounted fire extinguisher. No active fire. " + diagnosis.get("technical_analysis", "")
    return diagnosis

_NORMAL_GAUGE_PATTERNS = [
    re.compile(r'gauge\s+(?:is\s+)?pointing\s+(?:to|at|exactly|in\s+the)', re.IGNORECASE),
    re.compile(r'dipstick\s+(?:showing|between|level\s+between)', re.IGNORECASE),
    re.compile(r'(?:in\s+the\s+)?(?:green|normal)\s+zone', re.IGNORECASE),
]
_UNSAFE_KW_FOR_REASSURANCE = ["fire", "aag", "smoke", "dhuan", "leak", "tapak", "overheat", "garam", "sparks", "injury", "hurt", "chot", "bleeding"]

def _guard3_reassurance_visual_gauge(diagnosis, problem_text, machine_type, machine_label):
    if not _has_visual_context_text(problem_text): return None
    if not any(p.search(problem_text or "") for p in _NORMAL_GAUGE_PATTERNS): return None
    if any(kw in (problem_text or "").lower() for kw in _UNSAFE_KW_FOR_REASSURANCE): return None
    logger.info("🛡️ GUARD 3 NON-DESTRUCTIVE OVERRIDE: Normal visual gauge")
    diagnosis["status"] = "success"
    diagnosis["_reassurance_override"] = True
    return diagnosis

def _deduplicate_steps(steps: List[dict]) -> List[dict]:
    if not steps: return []
    action_verbs_re = re.compile(
        r'\b(check|inspect|look|clean|tighten|remove|replace|install|adjust|push|pull|turn|scan|tap|press|fill|drain|test|measure|hold)\b',
        re.IGNORECASE
    )
    seen, unique = set(), []
    for step in steps:
        req_part = step.get("required_part")
        if not req_part:
            unique.append(step)
            continue
        text_en = step.get("text_en") or ""
        m = action_verbs_re.search(text_en)
        action = m.group(1).lower() if m else ""
        vis_cue = step.get("visual_cue") or ""
        area_hint = step.get("area_hint") or ""
        sig = f"{req_part}:{vis_cue}:{area_hint}:{action}"
        if sig not in seen:
            seen.add(sig)
            unique.append(step)
    return unique

def _derive_unsafe_scene(problem_text: str, technical_analysis: str, has_any_visual: bool) -> tuple:
    pt = problem_text if problem_text else ""
    ta = technical_analysis if technical_analysis else ""
    combined = (pt + " " + ta).lower()
    clean_combined = combined.replace("fire extinguisher", "").replace("fire cylinder", "")
    injury_kw        = ["hurt", "chot", "cut", "kat gaya", "injured", "bleeding", "electric shock", "current laga"]
    active_hazard_kw = ["fire", "aag", "sparks flying", "smoke billowing", "oil spray", "fuel spray", "aag lagi", "jal raha", "sparks"]
    moving_near_kw   = ["belt moving", "ghoom raha", "spinning near", "rotating near", "hand near", "caught in"]
    if any(kw in clean_combined for kw in injury_kw):
        return True, "Injury/Shock risk detected. Stop immediately and ensure area is safe."
    if any(kw in clean_combined for kw in active_hazard_kw):
        return True, "Active hazard suspected (fire/spark/spray). Do not proceed until resolved."
    if any(kw in clean_combined for kw in moving_near_kw):
        return True, "Moving parts near inspection area. Ensure all rotation has stopped before inspection."
    if "dangerous workaround" in clean_combined:
        return True, "Dangerous mechanical workaround detected. Do not operate machine."
    return False, ""

def _fallback_escalation_response(machine_type, machine_label, problem_text, error_reason):
    return _build_escalation_response(
        machine_type=machine_type,
        machine_label=machine_label,
        problem_text=problem_text,
        technical_analysis=f"Unable to diagnose: {error_reason}",
        safety_warnings_en=["Automatic diagnosis unavailable.", "Consult a certified technician."],
        safety_warnings_hi=["स्वचालित निदान उपलब्ध नहीं है।", "प्रमाणित तकनीशियन से संपर्क करें।"],
        rag_source=RagSource.ERROR,
        problem_identified=problem_text,
    )


# ── FIX 6: Verification Gate ──────────────────────────────────────────────────

def _extract_risk_level_from_rag(rag_context: str) -> str:
    """Extract the highest risk level seen across chunks in the RAG context."""
    if not rag_context:
        return "LOW"
    # Scan all Risk: fields — take the highest
    risks = re.findall(r'\|\s*Risk:\s*(\w+)', rag_context)
    order = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
    if not risks:
        return "LOW"
    return max(risks, key=lambda r: order.get(r.upper(), 0)).upper()


async def _run_verification_gate(
    machine_type: str,
    problem_text: str,
    rag_context: str,
    risk_level: str,
    visual_frames: Optional[List[bytes]],
    language: str,
) -> Optional[dict]:
    """
    FIX 6: Verification gate — verify machine type and component existence
    before committing to the final repair steps.

    Runs only when:
      • risk_level is HIGH or CRITICAL
      • at least one visual frame is available
      • the diagnosis is not already escalated

    Returns None if verification passes or is skipped.
    Returns a "need_verification" dict if verification fails (< threshold confidence).
    """
    if risk_level not in _VERIFICATION_RISK_LEVELS:
        logger.debug("Verification gate SKIP: risk_level=%s", risk_level)
        return None

    if not visual_frames:
        logger.debug("Verification gate SKIP: no visual frames available")
        return None

    # Use the first frame for verification
    frame_bytes = visual_frames[0]

    # Extract the first step's required_part from the RAG context as the target
    # (best proxy for "what we're about to repair")
    required_part_match = re.search(r'"required_part"\s*:\s*"([^"]+)"', rag_context)
    required_part = required_part_match.group(1) if required_part_match else "main_component"

    # Extract area hint
    area_hint_match = re.search(r'"area_hint"\s*:\s*"([^"]+)"', rag_context)
    area_hint = area_hint_match.group(1) if area_hint_match else "engine_compartment"

    logger.info(
        "FIX 6: Running verification gate for %s | part=%s | risk=%s",
        machine_type, required_part, risk_level,
    )

    try:
        verification = await verify_step_with_gemini(
            image_bytes     = frame_bytes,
            step_text       = f"Verify this is the correct {machine_type} showing the {required_part}",
            required_part   = required_part,
            area_hint       = area_hint,
            machine_type    = machine_type,
            problem_context = problem_text,
            attempt_count   = 1,
            language        = language,
        )

        v_status = verification.get("status", "unclear")
        v_conf   = float(verification.get("confidence", 0.0))

        if v_status == "unsafe":
            logger.warning("FIX 6: Verification found UNSAFE condition — blocking repair")
            return {
                "status": "need_verification",
                "verification_failed": True,
                "verification_reason": "unsafe_condition_detected",
                "message_en": "An unsafe condition was detected in the camera image. Do not proceed with repairs. Ensure the area is safe first.",
                "message_hi": "कैमरे में असुरक्षित स्थिति दिखी। मरम्मत शुरू न करें। पहले क्षेत्र को सुरक्षित करें।",
                "ai_observation": verification.get("ai_observation", ""),
                "steps": [],
                "machine_type": machine_type,
            }

        if v_conf < _VERIFICATION_CONFIDENCE_THRESHOLD:
            logger.warning(
                "FIX 6: Verification confidence too low (%.2f < %.2f) — need verification",
                v_conf, _VERIFICATION_CONFIDENCE_THRESHOLD,
            )
            return {
                "status": "need_verification",
                "verification_failed": True,
                "verification_reason": "low_confidence",
                "confidence": round(v_conf, 3),
                "message_en": (
                    "I cannot confidently verify the machine type and component from the camera image. "
                    "Please ensure you are pointing the camera at the correct machine and component "
                    "before proceeding with repairs. Need verification before continuing."
                ),
                "message_hi": (
                    "मैं कैमरे की छवि से मशीन और पुर्जे की पुष्टि नहीं कर सका। "
                    "कृपया सुनिश्चित करें कि कैमरा सही मशीन और पुर्जे पर है। "
                    "आगे बढ़ने से पहले सत्यापन आवश्यक है।"
                ),
                "ai_observation": verification.get("ai_observation", ""),
                "steps": [],
                "machine_type": machine_type,
            }

        logger.info(
            "FIX 6: Verification PASSED: part=%s conf=%.2f status=%s",
            required_part, v_conf, v_status,
        )
        return None  # Verification passed — proceed normally

    except Exception as exc:
        logger.warning("FIX 6: Verification gate error (non-fatal): %s", exc)
        return None  # On error, allow through (don't block farmer)


# MIGRATED: Groq format normalization
def _normalize_status(diagnosis: dict) -> dict:
    """
    Normalize Groq/Gemini status to stable top-level lowercase.  # MIGRATED: Groq format normalization

    Supports:
      top-level status
      nested solution.status
      uppercase variants

    Maps:
      diagnose -> success
    """  # MIGRATED: Groq format normalization
    if not isinstance(diagnosis, dict):  # MIGRATED: Groq format normalization
        return diagnosis  # MIGRATED: Groq format normalization

    status = diagnosis.get("status")  # MIGRATED: Groq format normalization

    if not status:  # MIGRATED: Groq format normalization
        solution = diagnosis.get("solution", {})  # MIGRATED: Groq format normalization
        if isinstance(solution, dict):  # MIGRATED: Groq format normalization
            status = solution.get("status")  # MIGRATED: Groq format normalization

    status = str(status or "").strip().lower()  # MIGRATED: Groq format normalization

    if status == "diagnose":  # MIGRATED: Groq format normalization
        status = "success"  # MIGRATED: Groq format normalization

    if status:  # MIGRATED: Groq format normalization
        diagnosis["status"] = status  # MIGRATED: Groq format normalization

    # The top-level status above is the source of truth, but the LLM
    # writes its own (unnormalized) status into solution.status too, and
    # nothing previously synced them — so a raw "diagnose" literal (not a
    # valid value anywhere downstream) could sit right next to a correctly
    # normalized "success" in the same response. Force solution.status to
    # match, so there is exactly one status value in this object, not two.
    solution = diagnosis.get("solution")  # MIGRATED: Groq format normalization
    if isinstance(solution, dict) and status:  # MIGRATED: Groq format normalization
        solution["status"] = status  # MIGRATED: Groq format normalization

    return diagnosis  # MIGRATED: Groq format normalization

def _contains_actionable_guidance(diagnosis: dict) -> bool:
    """True if the _post_process_diagnosis contains concrete repair guidance (not just text)."""
    solution = diagnosis.get("solution", {})
    return bool(
        solution.get("steps")
        or solution.get("tools_needed")
        or solution.get("parts")
        or solution.get("verification_steps")
    )

async def _reword_call_llm(prompt: str) -> str:
    """Plain-text (non-JSON) LLM call used only by the jargon guard's
    single-field targeted reword retry. Deliberately not JSON_CONFIG —
    we want one plain rewritten sentence back, not a structured object.
    """
    response = await asyncio.to_thread(
        lambda: groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )
    )
    return response.choices[0].message.content


async def _post_process_diagnosis(
    diagnosis: dict,
    machine_type: str,
    machine_label: str,
    problem_text: str,
    rag_context: str,
    visual_frames: Optional[List[bytes]],
    context_quality: str,
) -> dict:

    # MIGRATED: Groq format normalization — normalize before any routing logic
    diagnosis = _normalize_status(diagnosis)  # MIGRATED: Groq format normalization

    # 1. NON-DESTRUCTIVE ROUTING OVERRIDES (Safety Injectors)
    g1 = _guard1_fire_extinguisher(diagnosis, problem_text, rag_context, machine_type, machine_label)
    g3 = _guard3_reassurance_visual_gauge(diagnosis, problem_text, machine_type, machine_label)

    if g1: diagnosis = g1
    if g3: diagnosis = g3

    # 2. Schema integrity
    if "solution" not in diagnosis or not isinstance(diagnosis.get("solution"), dict):
        diagnosis["solution"] = {}

    if diagnosis.get("status") == "escalate" and not diagnosis.get("rag_source"):
        diagnosis["rag_source"] = "escalation"

    # 3. Normalize step field names + deduplication
    # The LLM (especially fallback models) may return steps with non-standard
    # field names like "description"/"warning" instead of "text_en"/"safety_warning".
    # Normalize them so Flutter always sees a consistent schema.
    steps = diagnosis["solution"].get("steps", [])

    # ── Readability backstop ────────────────────────────────────────────
    # The prompt forbids chunk-header arrows ("→"/"->") in action/description,
    # but LLM compliance on this isn't guaranteed every run (observed: "Tighten
    # the shaft → coupling → motor connection" slipping through). This is a
    # deterministic code-level fix, not a prompt request — it always fires,
    # regardless of what the LLM returns, so a farmer never sees a raw
    # manual-heading arrow chain. Pure text cleanup only: no rewording, no
    # semantic changes, nothing that could alter what a step actually says.
    _ARROW_RE = re.compile(r"\s*(?:\u2192|->)\s*")

    def _strip_arrows(text: str) -> str:
        if not text:
            return text
        return _ARROW_RE.sub(" and ", text).strip()

    _normalized_steps = []
    for _s in steps:
        if not isinstance(_s, dict):
            continue
        _ns = dict(_s)
        for _field in ("action", "description", "text_en", "text_hi"):
            if _ns.get(_field):
                _ns[_field] = _strip_arrows(_ns[_field])

        # NOTE: jargon guard intentionally does NOT run here. It runs once,
        # later, on the truly final step list — see below, after
        # _deduplicate_steps() and validate_procedure() — because
        # validate_procedure() can replace diagnosis["solution"]["steps"]
        # wholesale (its own safe_steps), and any check done here would
        # miss content introduced or reshaped after this point.

        # Build text_en from action + description without destroying original fields.
        # action = short imperative (good for title/speech), description = explanation.
        # Concatenate both so the farmer sees the full instruction.
        _action = _ns.get("action", "").strip()
        _desc = _ns.get("description", "").strip()

        if "text_en" not in _ns:
            # Format snake_case action for display
            _display = _action.replace("_", " ") if "_" in _action and " " not in _action else _action
            if _display and _desc:
                _ns["text_en"] = f"{_display}\n{_desc}"
            elif _display:
                _ns["text_en"] = _display
            elif _desc:
                _ns["text_en"] = _desc
            elif _action:
                _ns["text_en"] = _action
            elif _desc:
                _ns["text_en"] = _desc

        # Map other common LLM field name variations to canonical names
        if "description_hi" in _ns and "text_hi" not in _ns:
            _ns["text_hi"] = _ns.get("description_hi", "")
        if "warning" in _ns and "safety_warning" not in _ns:
            _ns["safety_warning"] = _ns.get("warning", "")
        if "part" in _ns and "required_part" not in _ns:
            _ns["required_part"] = _ns.get("part", "")
        # Infer step_type if LLM didn't provide it — diagnosis owns this classification
        if "step_type" not in _ns:
            if not _ns.get("required_part") or _ns.get("required_part") in ("none", "unknown", ""):
                _ns["step_type"] = "safety"
            else:
                _ns["step_type"] = "inspection"
        if "area_hint" not in _ns or not _ns.get("area_hint"):
            _ns["area_hint"] = "engine_compartment"

        _normalized_steps.append(_ns)
    # DEBUG: Verify filter will run
    logger.info(f"🔍 DEBUG: About to filter {len(_normalized_steps)} steps")
    for i, s in enumerate(_normalized_steps):
        logger.info(f"🔍   Step {i}: type={s.get('step_type')} part={s.get('required_part')}")

    # ── Filter safety steps handled by Flutter SafetyGate ─────────────────
    # This runs AFTER the normalization loop, filtering the complete list.
    _SHUTDOWN_PARTS = {"ignition_key", "power_switch", "mcb", "circuit_breaker", 
                       "main_isolator", "battery_terminal", "fuel_valve", "machine_part"}
    
    _steps_before = len(_normalized_steps)
    _normalized_steps = [
        s for s in _normalized_steps
        if not (
            s.get("step_type") == "safety"
            and s.get("required_part", "") in _SHUTDOWN_PARTS
        )
    ]
    if len(_normalized_steps) < _steps_before:
        logger.info(f"🛡️  Removed %d safety step(s) — Flutter SafetyGate handles pre-AR safety",
            _steps_before - len(_normalized_steps))
    
    steps = _normalized_steps
    diagnosis["solution"]["steps"] = _deduplicate_steps(steps)
    if diagnosis.get("status") != "escalate":
        try:
            raw_steps  = diagnosis["solution"].get("steps", [])
            risk_level = diagnosis.get("risk_level") or "MEDIUM"
            tools_list = diagnosis["solution"].get("tools_needed", [])
            validation = validate_procedure(
                steps        = raw_steps,
                machine_type = machine_type,
                risk_level   = risk_level,
                tools_list   = tools_list,
                safe_inject  = False,
            )

            # Always apply safe_steps — even in Path A this is the correct list
            # (auto-injected shutdown step is now Flutter-compatible per FIX A2).
            diagnosis["solution"]["steps"] = validation.safe_steps

            # Log all issues unconditionally — not gated by passed flag.
            logger.info("FIX 5: %s", validation.summary())
            for issue in validation.issues:
                level = logging.WARNING if issue.severity in ("CRITICAL", "HIGH") else logging.DEBUG
                logger.log(level, "  [%s] %s: %s", issue.severity, issue.rule, issue.detail[:80])

            # Path B: unresolvable CRITICALs — block the response.
            # This is the enforcement gate that was previously missing.
            unresolvable_criticals = [
                i for i in validation.issues
                if i.severity == "CRITICAL" and not i.auto_fix
            ]
            if unresolvable_criticals:
                logger.error(
                    "FIX 5: BLOCKING dangerous steps for %s — %d unresolvable CRITICAL issue(s): %s",
                    machine_type,
                    len(unresolvable_criticals),
                    [i.rule for i in unresolvable_criticals],
                )
                diagnosis["status"]  = "escalate"
                diagnosis["solution"]["steps"] = []
                critical_detail_msgs = [i.detail for i in unresolvable_criticals]
                existing_en = diagnosis["solution"].get("safety_warnings_en", [])
                diagnosis["solution"]["safety_warnings_en"] = critical_detail_msgs + existing_en
                # Preserve machine_label so Flutter can render the escalation card correctly.
                diagnosis.setdefault("machine_label", machine_label)

            # Path B also: surface HIGH non-fixable issues as warnings (no blocking,
            # but they must be visible — not just in logs).
            elif not validation.passed:
                # HIGH issues only (CRITICALs handled above, WARNINGs don't block)
                high_msgs = [
                    i.detail for i in validation.issues
                    if i.severity == "HIGH" and not i.auto_fix
                ]
                if high_msgs:
                    existing_en = diagnosis["solution"].get("safety_warnings_en", [])
                    diagnosis["solution"]["safety_warnings_en"] = high_msgs + existing_en

        except Exception as exc:
            logger.warning("FIX 5: Procedure validation failed (non-fatal): %s", exc)
    # ────────────────────────────────────────────────────────────────────────

    # ── Jargon backstop — runs LAST, on the true final step list ───────────
    # Deliberately placed after dedup and validate_procedure() (not during
    # the earlier normalization loop): validate_procedure() can replace
    # diagnosis["solution"]["steps"] wholesale via its own safe_steps, so
    # checking any earlier than this would miss content it introduces or
    # reshapes. Checks only the first sentence of action/description (a
    # technical name later in the text, once the part is visually
    # introduced, is fine) and attempts a bounded reword retry — never a
    # full regeneration — on violation. See agent/jargon_guard.py.
    for _step in diagnosis["solution"].get("steps", []):
        if not isinstance(_step, dict):
            continue
        _changed = False
        for _field in ("action", "description"):
            if _step.get(_field):
                _fixed = await apply_jargon_guard(_step[_field], _reword_call_llm, label=_field)
                if _fixed != _step[_field]:
                    _step[_field] = _fixed
                    _changed = True
        if _changed:
            # Keep text_en in sync with the (possibly reworded) action/
            # description — it was originally built as "{action}\n{description}"
            # during normalization and would otherwise go stale.
            _action = (_step.get("action") or "").strip()
            _desc = (_step.get("description") or "").strip()
            if _action and _desc:
                _step["text_en"] = f"{_action}\n{_desc}"
            elif _action or _desc:
                _step["text_en"] = _action or _desc
    # ────────────────────────────────────────────────────────────────────────

    # ── Step ID assignment — SINGLE SOURCE OF TRUTH ─────────────────────────
    # This runs once, here, after every mutation that can add/remove/reorder
    # steps (dedup + procedure_validator's safe_steps injection) has already
    # happened. Every downstream consumer — Flutter's SolutionData, the
    # agent's RepairPlan/RepairPlanStep — must trust diagnosis["solution"]
    # ["steps"][i]["step_id"] as final. No fallback generation, no silent
    # patching anywhere further down the pipeline (session creation, the
    # agent's step-advancement logic, etc.) — if it's missing there, that's
    # a bug in THIS function, not something to paper over downstream.
    _final_steps = diagnosis["solution"].get("steps", [])
    for _idx, _step in enumerate(_final_steps):
        if isinstance(_step, dict):
            _step["step_id"] = f"s{_idx + 1}"

    # Structural validation happens immediately after generation — not
    # thousands of lines later — so an invalid plan can never exist beyond
    # this point. If this raises, it means THIS function has a bug (the
    # loop above is the only thing that sets step_id, so blank/duplicate
    # ids here would mean the assignment itself is broken). This is a
    # backend defect, not a mechanical fault — it must propagate as
    # InvalidRepairPlan, NOT get wrapped in an escalation card telling the
    # farmer to "consult a certified mechanic." Nothing is wrong with their
    # equipment; something is wrong with this service. The caller of
    # generate_diagnosis_with_gemini() (main.py) must catch InvalidRepairPlan
    # separately from normal escalation handling and return a generic
    # service-error response.
    #
    # EXCEPTION: status="escalate" with an empty step list is a valid,
    # intentional LLM output (RULE 1/3/5/9 in the prompt all instruct
    # steps=[] on escalation) — not a defect, so it must not raise here.
    # If an escalate response DOES carry steps (RULE 4 maintenance case),
    # those steps still go through full structural validation below.
    _status_norm = str(diagnosis.get("status", "")).strip().lower()
    logger.info("STATUS=%s steps=%d", _status_norm, len(_final_steps))
    if not (_status_norm == "escalate" and len(_final_steps) == 0):
        validate_repair_plan_steps(_final_steps, context=f"machine={machine_type}")
    # ────────────────────────────────────────────────────────────────────────

    # ── C7: Weak-context grounding enforcement ─────────────────────────────
    # The LLM is instructed to escalate when context is weak, but that's a soft
    # prompt instruction. This is the code-level backstop: if retrieval quality
    # is weak AND the model returned actionable guidance anyway, force escalation
    # through the canonical factory. No in-place mutation — single return path.
    if context_quality == "weak" and _contains_actionable_guidance(diagnosis):
        logger.warning(
            "C7: Weak context with actionable guidance (status='%s', steps=%d) — "
            "forcing escalation via factory",
            diagnosis.get("status"),
            len(diagnosis.get("solution", {}).get("steps", [])),
        )
        return _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=(
                "The repair manual does not contain enough information to "
                "safely guide this repair. Consult a certified mechanic."
            ),
            safety_warnings_en=[
                "The repair manual does not contain enough information to "
                "safely guide this repair. Consult a certified mechanic."
            ],
            safety_warnings_hi=[
                "मरम्मत मैनुअल में इस समस्या के लिए पर्याप्त जानकारी नहीं है। "
                "कृपया प्रमाणित मैकेनिक से संपर्क करें।"
            ],
            rag_source=RagSource.NO_CONTEXT,
            problem_identified="Insufficient knowledge base coverage for safe repair guidance.",
        )

    # 4. Unsafe-scene flag
    if diagnosis.get("_reassurance_override") or diagnosis.get("_informational_override") or diagnosis.get("_electric_hazard_override"):
        pass
    else:
        technical_analysis = diagnosis.get("technical_analysis") or ""
        unsafe, unsafe_msg = _derive_unsafe_scene(
            problem_text or "", technical_analysis,
            bool(visual_frames) or _has_visual_context_text(problem_text or "")
        )
        _no_real_hazard_sources = {"no_context", "escalation", "guard_unknown_attachment", None}
        _hazard_keywords = ["fire", "aag", "shock", "bijli", "current laga", "blood", "bleeding", "hurt", "chot", "sparks", "jal raha", "aag lagi"]
        prob_lower = (problem_text or "").lower()
        if (unsafe and not diagnosis.get("_electric_hazard_override") and diagnosis.get("rag_source") in _no_real_hazard_sources and not any(kw in prob_lower for kw in _hazard_keywords)):
            unsafe, unsafe_msg = False, ""
        if "unsafe_scene_suspected" not in diagnosis:
            diagnosis["unsafe_scene_suspected"] = unsafe
            if unsafe:
                diagnosis["unsafe_scene_message"] = unsafe_msg

    # 5. Metadata
    if not diagnosis.get("rag_source"):
        diagnosis["rag_source"] = "RAG+Gemini" if rag_context else "Gemini-only"
    diagnosis["machine_label"] = machine_label

    diagnosis = _enrich_industrial_response(diagnosis, rag_context, problem_text)

    return diagnosis


# ─────────────────────────────────────────────────────────────────────────────
# INDUSTRIAL RESPONSE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_industrial_response(
    diagnosis: dict,
    rag_context: str,
    problem_text: str,
) -> dict:
    sol = diagnosis.get("solution", {})

    if not diagnosis.get("confidence_label"):
        if rag_context:
            m = re.search(r"\[Retrieval Confidence:\s*(\w+)\]", rag_context)
            diagnosis["confidence_label"] = m.group(1) if m else "MEDIUM"
        else:
            diagnosis["confidence_label"] = "INSUFFICIENT"

    if not sol.get("verification_steps"):
        steps = sol.get("steps", [])
        if steps and diagnosis.get("status") not in ("escalate", "unsafe"):
            last_step = steps[-1] if steps else {}
            part = last_step.get("part", last_step.get("visual_cue", "the repaired part"))
            sol["verification_steps"] = [
                f"After completing repairs, run the machine for 2–3 minutes at idle.",
                f"Check {part} for leaks, unusual noise, or heat.",
                f"Verify normal operation: expected output should return to baseline.",
                "If the original symptom persists, escalate to a certified mechanic.",
            ]

    if not sol.get("escalation_conditions"):
        escalation_lines = re.findall(r"ESCALATE_IF:\n(.+?)(?=\n[A-Z]|\Z)", rag_context or "", re.DOTALL)
        if escalation_lines:
            sol["escalation_conditions"] = [l.strip() for l in escalation_lines[:3] if l.strip()]
        else:
            sol["escalation_conditions"] = [
                "Fault persists after following all repair steps.",
                "Any step involving electrical disassembly or internal component replacement.",
                "Machine shows signs of structural damage or unusual fluid loss.",
            ]

    if not sol.get("preventive_maintenance"):
        problem_lower = problem_text.lower()
        if any(kw in problem_lower for kw in ["bearing", "seal", "oil", "grease", "lubric"]):
            tip = "Check and replenish lubricant every 250 operating hours or as per manual schedule."
        elif any(kw in problem_lower for kw in ["belt", "drive", "coupling"]):
            tip = "Inspect drive belts for wear and correct tension every 100 hours."
        elif any(kw in problem_lower for kw in ["filter", "air", "fuel"]):
            tip = "Replace fuel and air filters at the manufacturer's recommended interval."
        elif any(kw in problem_lower for kw in ["battery", "electric", "wiring"]):
            tip = "Clean battery terminals monthly and check wiring insulation for cracks."
        else:
            tip = "Follow the manufacturer's periodic maintenance schedule to prevent recurrence."
        sol["preventive_maintenance"] = [tip]

    steps = sol.get("steps", [])
    issues: List[str] = []
    if not steps and diagnosis.get("status") == "success":
        issues.append("WARNING: status=success but steps list is empty.")
    if diagnosis.get("status") == "success" and steps:
        if not any(s.get("required_part") or s.get("text_en") for s in steps):
            issues.append("WARNING: steps present but missing required_part/text_en fields.")
    if not sol.get("safety_warnings_en"):   # covers None and []
        sol["safety_warnings_en"] = ["Follow all standard safety precautions."]
    if issues:
        logger.warning("Diagnosis verification issues: %s", issues)
        diagnosis["_verification_issues"] = issues

    diagnosis["solution"] = sol
    return diagnosis


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def generate_diagnosis_with_gemini(
    machine_type: str,
    problem_text: str,
    language: str = "en",
    rag_context: str = "",
    knowledge_base: str = "",
    visual_frames: Optional[List[bytes]] = None,
    router_symptoms: Optional[List[str]] = None,
    router_confidence: float = 1.0,
    clarification_round: int = 0,
    vector_db=None,
    top_score: float = 0.0,
) -> dict:
    logger.info(
        f"🧠 Diagnosis v11.0: machine={machine_type}, "
        f"problem='{problem_text[:60]}...', "
        f"rag={'YES' if rag_context else 'NO'}, "
        f"visual_text={'YES' if _has_visual_context_text(problem_text) else 'NO'}"
    )

    visual_hash = ""
    if visual_frames:
        mid_frame = visual_frames[len(visual_frames) // 2]
        visual_hash = hashlib.md5(mid_frame).hexdigest()[:8]
    profile       = get_profile_or_default(machine_type)
    machine_label = profile.label_en

    if top_score <= 0.0 and (rag_context or knowledge_base):
        score_match = re.search(r'Relevance:\s*([\d.]+)', rag_context or knowledge_base)
        if score_match:
            top_score = float(score_match.group(1))
        else:
            top_score = 0.60
    context_quality = "strong" if top_score >= _RAG_STRICT_THRESHOLD else "weak"

    cache_key = generate_cache_key("diag_v302", machine_type, f"{problem_text}|vhash:{visual_hash}", language)
    cached = get_cached_response(cache_key)
    if cached:
        try:
            processed = await _post_process_diagnosis(
                cached, machine_type, machine_label, problem_text,
                rag_context or knowledge_base, visual_frames,
                context_quality=context_quality,
            )
            processed["cache_hit"] = True
            return processed
        except InvalidRepairPlan as exc:
            # The cached plan re-validated as broken (e.g. cached before
            # this validation existed). This is a backend defect, not a
            # mechanical fault — do NOT wrap it in an escalation card.
            # The raw cached entry was itself never re-validated on write,
            # so it's not necessarily safe either; surface this loudly and
            # let it propagate so the API layer can return a service error
            # instead of silently serving a plan we know is broken.
            logger.error(
                "❌ Cached repair plan failed validation for machine=%s: %s",
                machine_type, exc,
            )
            raise
        except Exception as exc:
            logger.warning("Cache post-processing failed, returning raw cached plan: %s", exc)
            cached["cache_hit"] = True
            return cached
    allowed_areas = " | ".join(get_allowed_area_ids(machine_type))
    parts_list    = get_compact_parts_list(machine_type)
    safety_kw     = get_compact_safety_keywords(machine_type)

    effective_rag = rag_context or knowledge_base

    trigger = _should_hard_escalate(problem_text, effective_rag)
    if trigger:
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=f"Pre-call guard triggered: '{trigger}'. No input to process.",
            safety_warnings_en=["Automatic diagnosis unavailable: no input received."],
            safety_warnings_hi=["स्वचालित निदान उपलब्ध नहीं है: कोई इनपुट प्राप्त नहीं हुआ।"],
            rag_source=RagSource.PRE_CALL,
            problem_identified="Empty or invalid input.",
        )
        cache_response(cache_key, result)
        return result

    g5 = _guard5_empty_rag_maintenance(problem_text, effective_rag, machine_type, machine_label)
    if g5 is not None:
        cache_response(cache_key, g5)
        return g5

    g_unknown = _guard_unknown_attachment(problem_text, machine_type, machine_label)
    if g_unknown is not None:
        cache_response(cache_key, g_unknown)
        return g_unknown

    g_pre_electric = _guard_electric_hazard(problem_text, machine_type)
    if g_pre_electric is not None:
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=g_pre_electric["technical_analysis"],
            safety_warnings_en=g_pre_electric["safety_warnings_en"],
            safety_warnings_hi=g_pre_electric["safety_warnings_hi"],
            rag_source=RagSource.ELECTRIC_HAZARD,
            problem_identified=g_pre_electric["technical_analysis"],
            metadata={
                "unsafe_scene_suspected": True,
                "unsafe_scene_message": "Electric hazard detected. Ensure power is cut before contact.",
            },
        )
        cache_response(cache_key, result)
        return result

    g_pre_workaround = _guard_dangerous_workaround(problem_text)
    if g_pre_workaround is not None:
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=g_pre_workaround["technical_analysis"],
            safety_warnings_en=g_pre_workaround["safety_warnings_en"],
            safety_warnings_hi=g_pre_workaround["safety_warnings_hi"],
            rag_source=RagSource.WORKAROUND,
            problem_identified=g_pre_workaround["technical_analysis"],
            metadata={
                "unsafe_scene_suspected": True,
                "unsafe_scene_message": "Dangerous mechanical workaround detected. Do not operate machine.",
            },
        )
        cache_response(cache_key, result)
        return result
        
    g_emergency = _guard_emergency_hazard(problem_text)
    if g_emergency is not None:
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis=g_emergency["technical_analysis"],
            safety_warnings_en=g_emergency["safety_warnings_en"],
            safety_warnings_hi=g_emergency["safety_warnings_hi"],
            rag_source=RagSource.EMERGENCY_HAZARD,
            problem_identified=g_emergency["technical_analysis"],
        )
        cache_response(cache_key, result)
        return result

    # FIX 4: Config-driven threshold — skip multihop when primary retrieval is strong.
    # Saves 2 Gemini calls per request for high-confidence queries.
    _MULTIHOP_MIN_SCORE: float = float(os.environ.get("AGRIFIX_MULTIHOP_MIN_SCORE", "0.60"))
    
    symptoms = router_symptoms or []
    n_chunks_estimate = effective_rag.count("[Source:") if effective_rag else 0
    
    if top_score > _MULTIHOP_MIN_SCORE:
        logger.info(
            "FIX 4: Multihop skipped — score=%.3f > threshold=%.2f (fast-path)",
            top_score, _MULTIHOP_MIN_SCORE,
        )
        chain = None
    else:
        chain = await run_diagnostic_chain(
            machine_type       = machine_type,
            symptoms           = symptoms,
            rag_context        = effective_rag,
            router_confidence  = router_confidence,
            n_rag_chunks       = n_chunks_estimate,
        )
        if chain and chain.enriched_query and vector_db is not None and chain.chain_ok is not False:
            try:
                from rag import retrieve_with_confidence as _retrieve
                rag_v2, score_v2, n_v2 = _retrieve(vector_db, chain.enriched_query, machine_type)
                if score_v2 > top_score and rag_v2:
                    logger.info(
                        "FIX 3: Multi-hop re-retrieval improved score %.3f → %.3f (%d chunks)",
                        top_score, score_v2, n_v2,
                    )
                    effective_rag = rag_v2
                    top_score     = score_v2
                    context_quality = "strong" if top_score >= _RAG_STRICT_THRESHOLD else "weak"
            except Exception as exc:
                logger.warning("FIX 3: Multi-hop re-retrieval failed: %s", exc)
    # ────────────────────────────────────────────────────────────────────────

    if not effective_rag:
        logger.warning(f"⚠️ No RAG context for [{machine_type}] — returning safe no-context escalation")
        result = _build_escalation_response(
            machine_type=machine_type,
            machine_label=machine_label,
            problem_text=problem_text,
            technical_analysis="No relevant manual excerpts found for this query.",
            safety_warnings_en=[_NO_CONTEXT_ESCALATION_EN],
            safety_warnings_hi=[_NO_CONTEXT_ESCALATION_HI],
            rag_source=RagSource.NO_CONTEXT,
            problem_identified=problem_text,
        )
        cache_response(cache_key, result)
        return result

    # ── FIX 6: Pre-LLM Verification Gate ─────────────────────────────────────
    # Verify machine type and component before committing to repair steps.
    # Only fires for HIGH/CRITICAL risk when visual frames are available.
    risk_level_from_rag = _extract_risk_level_from_rag(effective_rag)
    verification_block = await _run_verification_gate(
        machine_type   = machine_type,
        problem_text   = problem_text,
        rag_context    = effective_rag,
        risk_level     = risk_level_from_rag,
        visual_frames  = visual_frames,
        language       = language,
    )
    if verification_block is not None:
        logger.warning("FIX 6: Verification gate BLOCKED repair: %s", verification_block.get("verification_reason"))
        cache_response(cache_key, verification_block)
        return verification_block
    # ────────────────────────────────────────────────────────────────────────

    prompt = _build_strict_grounding_prompt(
        machine_type      = machine_type,
        machine_label     = machine_label,
        problem_text      = problem_text,
        rag_context       = effective_rag,
        allowed_areas     = allowed_areas,
        parts_list        = parts_list,
        safety_keywords   = safety_kw,
        language          = language,
        has_visual_frames = bool(visual_frames),
        context_quality   = context_quality,
        top_score         = top_score,
        router_symptoms   = symptoms,
    )
    try:
        response = await asyncio.to_thread(
            lambda: groq_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **JSON_CONFIG,
            )
        )
        raw_text = response.choices[0].message.content
        diagnosis = repair_json(raw_text)
        if not isinstance(diagnosis, dict):
            raise ValueError(
                f"LLM returned non-dict JSON ({type(diagnosis).__name__}): "
                f"{str(diagnosis)[:80]}"
            )

        diagnosis = await _post_process_diagnosis(
            diagnosis, machine_type, machine_label, problem_text, effective_rag, visual_frames,
            context_quality=context_quality,
        )
        cache_response(cache_key, diagnosis)
        return diagnosis

    except InvalidRepairPlan as exc:
        # A structurally broken repair plan is a backend defect, not a
        # mechanical fault with the farmer's machine. Do NOT catch this
        # below and wrap it in _fallback_escalation_response() — that
        # would tell the farmer to "consult a certified mechanic" when
        # the actual problem is in this service. Let it propagate; the
        # API layer (main.py) must catch InvalidRepairPlan separately
        # and return a generic service-error response (e.g. HTTP 500).
        logger.error("❌ Invalid repair plan for machine=%s: %s", machine_type, exc)
        raise
    except Exception as exc:
        logger.error("❌ Diagnosis generation failed: %s", exc)
        return _fallback_escalation_response(machine_type, machine_label, problem_text, str(exc))