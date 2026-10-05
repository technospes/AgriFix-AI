"""
dense_narrative.py — AgriFixAR Dense Retrieval Narrative Generator v2.0
=======================================================================
Generates semantically rich, embedding-optimised narratives for ChromaDB
chunks. Each narrative includes:
  - Machine + symptom in multiple phrasings (aliasing)
  - Alternate farmer phrases (Hinglish + colloquial)
  - Technical synonyms for all fault modes
  - Failure progression (first → then → then)
  - Component references with context
  - Cause expansion into diagnostic sentences
  - Step instructions as natural-language sentences
  - Safety and tool context
  - Hinglish symptom mirrors for multilingual retrieval

Target: 150–280 words per narrative.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# ── Hinglish symptom mirror map ───────────────────────────────────────────────
_HINGLISH_SYMPTOM_MIRROR: Dict[str, str] = {
    "not starting":       "start nahi ho raha, chalu nahi ho raha, machine nahi chalti",
    "overheating":        "garam ho raha hai, bahut garam, overheat ho gaya, zyada temperature",
    "no water flow":      "paani nahi aa raha, pani band ho gaya, discharge nahi, paani band",
    "low discharge":      "paani kam aa raha, pressure kam hai, flow kam ho gaya",
    "vibration":          "hil raha hai, kaamp raha, vibration aa rahi, hila raha hai",
    "abnormal noise":     "awaaz aa rahi hai, khadkhadana, tak tak awaaz, gharrr awaaz, machine awaz kar rahi",
    "humming":            "gungunaahat, ghurr ghurr awaaz, motor gungunata hai, bhanbhanahat",
    "smoke":              "dhuan aa raha, dhuen nikal raha, jalney ki boo",
    "oil leak":           "tel tapak raha, oil tapak rahi, tel nikal raha",
    "power loss":         "dum nahi raha, kamzor ho gaya, load nahi le raha, power kam",
    "seized":             "jam gaya, ghoom nahi raha, shaft jam gaya, band ho gaya",
    "air lock":           "hawa maar raha, hawa bhar gaya, hawa phans gaya",
    "tripped breaker":    "MCB trip ho gaya, switch upar aa gaya, breaker gir gaya",
    "capacitor failure":  "capacitor kharab, capacitor phul gaya, capacitor fail",
    "bearing failure":    "bearing kharab, ghisghisana awaaz, bearing ghis gaya",
    "fuel problem":       "diesel nahi, fuel nahi aa raha, petrol khatam",
    "belt slip":          "belt phisal rahi, belt loose, belt dhili ho gayi",
    "overload":           "overload ho gaya, zyada load, motor overloaded",
    "single phasing":     "ek phase band, single phase ho gaya, phase gaya",
    "voltage drop":       "voltage kam, bijli weak, supply kam aa rahi",
    "impeller blocked":   "impeller jam gaya, impeller mein mitti, impeller band",
    "shaft broken":       "shaft toot gaya, shaft bend ho gaya, shaft kharab",
    "starter failure":    "starter nahi chal raha, starter fail, start nahi karta",
    "no spark":           "spark nahi aa raha, ignition nahi, current nahi",
    "clogged filter":     "filter band ho gaya, filter jam gaya, filter saaf karo",
}

# ── Technical synonym expansions ──────────────────────────────────────────────
_TECH_SYNONYMS: Dict[str, List[str]] = {
    "capacitor":       ["start capacitor", "run capacitor", "motor capacitor", "kondenser"],
    "bearing":         ["ball bearing", "roller bearing", "shaft bearing", "pillow block bearing"],
    "impeller":        ["pump impeller", "centrifugal impeller", "vane", "runner"],
    "relay":           ["starter relay", "contactor", "magnetic contactor", "overload relay"],
    "mcb":             ["miniature circuit breaker", "breaker", "circuit breaker", "switch"],
    "winding":         ["motor winding", "stator winding", "coil", "armature winding"],
    "shaft":           ["drive shaft", "motor shaft", "pump shaft", "rotor shaft"],
    "seal":            ["mechanical seal", "lip seal", "shaft seal", "o-ring seal"],
    "injector":        ["fuel injector", "nozzle", "atomizer", "spray nozzle"],
    "governor":        ["speed governor", "fuel governor", "throttle governor"],
    "alternator":      ["charging unit", "dynamo", "generator", "AC generator"],
    "glow plug":       ["heater plug", "pre-heat plug", "starting aid"],
    "foot valve":      ["non-return valve", "check valve", "suction valve", "NRV"],
    "primer":          ["primer bulb", "priming pump", "hand primer"],
    "pto":             ["power take-off", "PTO shaft", "auxiliary shaft"],
}

# ── Failure progression templates ─────────────────────────────────────────────
_FAULT_PROGRESSION: Dict[str, str] = {
    "capacitor failure": (
        "Initially the motor hums but shaft does not rotate. "
        "Then on repeated attempts, the motor trips the breaker. "
        "Finally the motor overheats and emits a burning smell."
    ),
    "bearing failure": (
        "First a grinding or squealing noise develops during operation. "
        "Then vibration increases and the shaft runs hot. "
        "Finally the bearing seizes and the machine stops completely."
    ),
    "overheating": (
        "First the machine loses power and speed drops. "
        "Then the motor body becomes too hot to touch. "
        "Finally the thermal overload trips and stops the machine."
    ),
    "air lock": (
        "First the pump runs but no water comes out. "
        "Then the motor may overspeed from running dry. "
        "Finally the mechanical seal overheats from lack of water lubrication."
    ),
    "fuel starvation": (
        "First the engine hunts or surges at idle. "
        "Then it loses power under load. "
        "Finally it stalls and cannot restart."
    ),
    "belt slip": (
        "First a squealing sound is heard at start-up. "
        "Then the driven component slows down under load. "
        "Finally the belt overheats, glazes, and breaks."
    ),
    "overload": (
        "First the motor slows under heavy load. "
        "Then the overload relay trips. "
        "Finally repeated trips cause winding overheating."
    ),
    "single phasing": (
        "First the motor hums but does not start or runs at reduced speed. "
        "Then it draws excessive current on the two live phases. "
        "Finally the winding burns out from unbalanced heating."
    ),
}


def _get_hinglish_mirrors(symptoms: List[str]) -> str:
    mirrors = []
    for sym in symptoms:
        sym_lower = sym.lower()
        for eng, hindi in _HINGLISH_SYMPTOM_MIRROR.items():
            if eng in sym_lower or any(word in sym_lower for word in eng.split()):
                if hindi not in mirrors:
                    mirrors.append(hindi)
                break
    return "; ".join(mirrors) if mirrors else ""


def _expand_tech_synonyms(component: str) -> str:
    c_lower = component.lower()
    for key, synonyms in _TECH_SYNONYMS.items():
        if key in c_lower:
            return f"{component} (also called: {', '.join(synonyms[:3])})"
    return component


def _get_failure_progression(causes: List[str], symptoms: List[str]) -> str:
    text = " ".join(c.lower() for c in causes + symptoms)
    for key, progression in _FAULT_PROGRESSION.items():
        if any(word in text for word in key.split()):
            return progression
    return ""


def dense_retrieval_narrative(proc: dict) -> str:
    """
    Convert a structured repair procedure into a DENSE retrieval narrative.
    Drop-in replacement for _generate_retrieval_narrative() in build_knowledge.py.
    Signature: (proc: dict) -> str

    FIX 7: The narrative now opens with a short PROBLEM: line that mirrors the
    `problem` metadata field stored in ChromaDB.  This ensures _metadata_match_score()
    gets a populated, direct symptom phrase (e.g. "submersible pump not starting
    after power cut") rather than a generic section heading.  The submersible_pump
    category had 33% accuracy because `problem` was empty or prose-heavy, killing
    the metadata signal entirely.
    """
    machine_family  = proc.get("machine_family", proc.get("machine_type", "machine"))
    machine_type    = proc.get("machine_type", "machine")
    system          = proc.get("system", "")
    component       = proc.get("component", "")
    symptoms        = proc.get("symptoms", [])
    aliases         = proc.get("symptoms_canonical", [])
    causes          = proc.get("causes", [])
    steps           = proc.get("step_sequence", [])
    warnings        = proc.get("safety_warnings", [])
    tools           = proc.get("required_tools", [])
    parts           = proc.get("part_numbers", [])
    escalate_if     = proc.get("escalate_if", "")
    severity        = proc.get("severity", "")
    env_notes       = proc.get("environmental_conditions", "")
    source          = proc.get("manual_source", "")

    machine_desc = f"{machine_family} {system}".strip() if system else machine_family
    if not machine_desc:
        machine_desc = machine_type.replace("_", " ")

    symptom_str = "; ".join(symptoms) if symptoms else "a reported fault"
    component_expanded = _expand_tech_synonyms(component) if component else ""

    # ── FIX 7: Problem summary line ───────────────────────────────────────────
    # Build a short, direct problem_summary that will be stored as the `problem`
    # metadata field.  _metadata_match_score() weights this field heavily; if it
    # is empty or generic the metadata signal is effectively dead.
    # Format: "<machine_family> <primary_symptom>" — max ~10 words, no fluff.
    primary_symptom = (symptoms[0] if symptoms else (aliases[0] if aliases else "fault"))
    problem_summary = f"{machine_family} {primary_symptom}".strip().lower()
    # Emit it as the very first line so it is indexed at the start of page_content
    block0 = f"PROBLEM: {problem_summary}"

    # ── Block 1: Problem statement with aliasing ──────────────────────────────
    b1_parts = []
    b1_parts.append(
        f"The {machine_desc} is experiencing: {symptom_str}."
    )
    if component_expanded:
        b1_parts.append(
            f"The affected component is the {component_expanded} of the {machine_family}."
        )

    alias_candidates = list(dict.fromkeys(aliases + [s for s in symptoms if s not in aliases]))
    if len(alias_candidates) > 1:
        b1_parts.append(
            f"This fault is also described as: {', '.join(alias_candidates[:6])}. "
            f"Farmers commonly report this as: machine not working, {alias_candidates[0].lower()}."
        )

    if severity:
        b1_parts.append(f"Fault severity: {severity}.")
    if env_notes:
        b1_parts.append(f"Environmental context: {env_notes}.")

    block1 = " ".join(b1_parts)

    # ── Block 2: Failure progression ─────────────────────────────────────────
    progression = _get_failure_progression(causes, symptoms)
    if progression:
        block2 = f"Typical failure progression: {progression}"
    else:
        block2 = ""

    # ── Block 3: Cause analysis ───────────────────────────────────────────────
    cause_sentences = []
    for i, cause in enumerate(causes[:5]):
        if i == 0:
            cause_sentences.append(
                f"The primary root cause is: {cause}. "
                "This is the first failure mode to investigate."
            )
        elif i == 1:
            cause_sentences.append(
                f"Secondary cause: {cause}. Check this if primary cause is not found."
            )
        else:
            cause_sentences.append(f"Also possible: {cause}.")
    block3 = " ".join(cause_sentences) if cause_sentences else (
        f"The exact cause requires inspection of the "
        f"{component or system or 'affected assembly'}."
    )

    # ── Block 4: Repair procedure ─────────────────────────────────────────────
    step_sentences = []
    for s in steps[:7]:
        instr   = (s.get("instruction") or "").strip()
        expect  = (s.get("expected_result") or "").strip()
        if_fail = (s.get("if_fail_then") or "").strip()
        sn      = s.get("step_number", "")
        if not instr:
            continue
        st = f"Step {sn}: {instr}"
        if expect:
            st += f" Expected result: {expect}."
        if if_fail:
            st += f" If this fails: {if_fail}."
        step_sentences.append(st)
    block4 = " ".join(step_sentences) if step_sentences else (
        f"Inspect the {component or 'assembly'} for visible damage, blockage, or wear."
    )

    # ── Block 5: Safety + tools + escalation ─────────────────────────────────
    safety_parts = []
    for w in warnings[:3]:
        safety_parts.append(f"Safety warning: {w}.")
    if escalate_if:
        safety_parts.append(f"Escalate to certified mechanic if: {escalate_if}.")
    if tools:
        safety_parts.append(
            f"Required tools: {', '.join(tools[:5])}. "
            "Do not attempt this repair without the correct tools."
        )
    if parts:
        safety_parts.append(f"Replacement parts that may be needed: {', '.join(parts[:4])}.")
    if source:
        safety_parts.append(f"Source: {source}.")
    block5 = " ".join(safety_parts)

    # ── Block 6: Hinglish mirrors ─────────────────────────────────────────────
    hinglish = _get_hinglish_mirrors(symptoms + aliases)
    if hinglish:
        block6 = (
            f"Kisaan is samasya ko in shabdon mein bata sakte hain: {hinglish}. "
            f"Yeh samasya {machine_family} mein {symptom_str} ki wajah se hoti hai. "
            f"Pump paani nahi de raha, machine start nahi ho rahi, gharrr awaaz — "
            f"yeh sab is {component or 'part'} ki kharaabi ke lakshan hain."
        )
    else:
        block6 = ""

    paragraphs = [block0, block1, block2, block3, block4, block5, block6]
    narrative = "\n\n".join(p for p in paragraphs if p.strip())
    narrative = re.sub(r" {2,}", " ", narrative).strip()
    return narrative

def _flatten_str_list(items: list) -> list:
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(str(x) for x in item if x)
        elif item:
            result.append(str(item))
    return result


def dense_retrieval_narrative_with_problem(proc: dict):
    """
    FIX 7 helper: returns (narrative_str, problem_summary_str) so that callers
    in build_knowledge.py can store problem_summary directly in the `problem`
    ChromaDB metadata field without re-parsing the narrative.

    Usage in build_knowledge.py:
        narrative, problem_meta = dense_retrieval_narrative_with_problem(proc)
        metadata["problem"] = problem_meta
        doc = Document(page_content=narrative, metadata=metadata)
    """
    machine_family = proc.get("machine_family", proc.get("machine_type", "machine"))
    symptoms       = proc.get("symptoms", [])
    aliases        = proc.get("symptoms_canonical", [])
    primary_symptom = (symptoms[0] if symptoms else (aliases[0] if aliases else "fault"))
    problem_summary = f"{machine_family} {primary_symptom}".strip().lower()
    narrative = dense_retrieval_narrative(proc)
    return narrative, problem_summary


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_proc = {
        "machine_family":     "electric water pump",
        "machine_type":       "water_pump",
        "system":             "motor and impeller assembly",
        "component":          "capacitor",
        "symptoms":           ["motor hums but does not rotate", "no water discharge"],
        "symptoms_canonical": ["not starting", "humming", "no flow"],
        "causes": [
            "failed start capacitor — insufficient starting torque",
            "seized impeller due to sand ingress",
            "single-phase supply fault",
        ],
        "step_sequence": [
            {
                "step_number": 1,
                "instruction": "Switch off MCB and isolate power supply.",
                "expected_result": "All indicator lights off.",
                "if_fail_then": "Do not proceed — contact electrician.",
            },
            {
                "step_number": 2,
                "instruction": "Discharge capacitor by bridging terminals with insulated screwdriver.",
                "expected_result": "Small spark confirms charge discharged.",
            },
            {
                "step_number": 3,
                "instruction": "Measure capacitor µF with multimeter. Compare to nameplate value.",
                "expected_result": "Reading within ±10% of rated value.",
                "if_fail_then": "Replace capacitor with same µF and voltage rating.",
            },
        ],
        "safety_warnings": [
            "Never touch capacitor terminals without discharging first.",
            "Ensure pump is isolated from power before opening motor cover.",
        ],
        "required_tools":  ["insulated screwdriver", "multimeter", "spanner set"],
        "part_numbers":    ["CAP-25UF-450V"],
        "escalate_if":     "Motor winding smells burnt or capacitor is physically swollen.",
        "severity":        "HIGH",
        "manual_source":   "Kirloskar_Pump_Service_Manual_v3.pdf",
    }

    narrative = dense_retrieval_narrative(sample_proc)
    wc = len(narrative.split())
    print(f"\n{'='*70}\nDENSE NARRATIVE ({wc} words):\n{'='*70}")
    print(narrative)
    print(f"\nTarget: 150-280 words. Got: {wc}")