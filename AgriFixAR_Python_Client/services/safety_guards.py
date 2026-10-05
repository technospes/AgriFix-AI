"""
services/safety_guards.py
Deterministic text-hazard guards — single source of truth for all endpoints.

Every endpoint that accepts free-text farmer input (/diagnose, /agent/next,
/verify_step, /locate_part) runs these guards before any LLM call.

Architecture:
    services/safety_guards.py       ← guard implementations + public API
           ↑
    main.py (all endpoints)         ← call run_text_hazard_checks() only
    services/diagnosis_service.py   ← imports guards from here
    agent/safety_rules.py           ← imports guards from here
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List

logger = logging.getLogger(__name__)


# ── Public API types ──────────────────────────────────────────────────────────

class GuardSeverity(IntEnum):
    """Ordered by severity — higher values indicate more urgent hazards."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GuardResult:
    """Structured result from hazard checks. Safe for external consumption."""
    blocked: bool
    severity: GuardSeverity
    technical_analysis: str
    safety_warnings_en: List[str] = field(default_factory=list)
    safety_warnings_hi: List[str] = field(default_factory=list)
    guard_name: str = ""

    @property
    def user_message_en(self) -> str:
        """Safe for display to farmers — first English warning."""
        return self.safety_warnings_en[0] if self.safety_warnings_en else "STOP. Unsafe action detected."

    @property
    def user_message_hi(self) -> str:
        """Safe for display to farmers — first Hindi warning."""
        return self.safety_warnings_hi[0] if self.safety_warnings_hi else "रुकें। असुरक्षित कार्रवाई का पता चला।"


# ── Guard: Dangerous workaround patterns ─────────────────────────────────────

_DANGEROUS_WORKAROUND_PATTERNS = [
    (re.compile(r'\brubber\s*band\b', re.IGNORECASE), "rubber band used as governor/throttle spring"),
    (re.compile(
        r'\bwire\s+(?:for|instead|as)\s+belt\b'
        r'|\bwire\s+(?:for|instead\s+of|as)\s+(?:a\s+)?belt\b'
        r'|\bwire\s*[\-\/]\s*belt'
        r'|\breplace\s+(?:the\s+)?belt\s+(?:with|by|for)\s+(?:a\s+)?wire\b'
        r'|\bwire\s+se\s+(?:belt|bandh|kaam\s+chala)\b'
        r'|\b(?:wire|taar)\s+(?:belt|v[\-\s]?belt)\s+(?:ki\s+jagah|ke\s+badle|laga)\b'
        r'|\btaar\s+(?:laga\s+diya|lagaya|bandh\s+kar)\b',
        re.IGNORECASE
    ), "wire used instead of belt"),
    (re.compile(r'\brope\s+(?:for|instead|as)\s+(?:drive\s+)?belt\b', re.IGNORECASE), "rope used as drive belt"),
    (re.compile(
        r'\bnail\s+(?:for|instead|as)\s+fuse\b'
        r'|\bnail\s+(?:for|instead\s+of|as)\s+(?:a\s+)?fuse\b'
        r'|\bfuse\s+(?:replace|swap).*?(?:with|for)\s+(?:a\s+)?nail\b',
        re.IGNORECASE
    ), "nail used as fuse"),
    (re.compile(r'\bfoil\s+(?:for|instead|as)\s+fuse\b', re.IGNORECASE), "foil used as fuse"),
    (re.compile(r'\bstring\s+(?:tied|for|as)\s+throttle\b', re.IGNORECASE), "string tied to throttle"),
    (re.compile(r'\bbypas(?:s|sing)\s+(?:safety|interlock|switch|sensor)\b', re.IGNORECASE), "safety bypass detected"),
]

_WORKAROUND_WARNINGS = {
    "rubber band used as governor/throttle spring": (
        "STOP. A rubber band on the throttle/governor is an extreme fire and overspeeding hazard. The engine cannot regulate speed and may overspeed to destruction. Do NOT start the engine. Replace with the correct OEM governor spring immediately.",
        "रुकें! थ्रॉटल/गवर्नर पर रबर बैंड लगाना बेहद खतरनाक है — इंजन बेकाबू होकर टूट सकता है। इंजन बिल्कुल मत चलाएं। तुरंत असली गवर्नर स्प्रिंग लगाएं।",
        "TO FIX: Remove the rubber band. Install the correct OEM governor spring between the governor arm and throttle linkage.",
    ),
    "wire used instead of belt": (
        "STOP. A wire cannot safely replace a drive belt — it will snap under load, become a projectile, or jam the pulleys causing serious injury. Do NOT operate the machine. Replace with the correct specification V-belt immediately.",
        "रुकें! तार बेल्ट की जगह नहीं ले सकता — यह टूटकर गंभीर चोट कर सकता है या पुली में फंस सकता है। मशीन बिल्कुल मत चलाएं। सही नाप की V-बेल्ट तुरंत लगाएं।",
        "TO FIX: Remove the wire. Loosen the tensioner pulley bolt, route the correct OEM V-belt over both pulleys, adjust tension until the belt deflects ~10mm when pressed firmly at centre.",
    ),
    "rope used as drive belt": (
        "STOP. A rope will slip, fray, and fail under load causing sudden power loss or injury. Do NOT operate. Replace with the correct specification V-belt.",
        "रुकें! रस्सी बेल्ट का काम नहीं करती — यह टूट सकती है। मशीन मत चलाएं। सही V-बेल्ट लगाएं।",
        "TO FIX: Remove the rope. Install the correct OEM V-belt.",
    ),
    "nail used as fuse": (
        "STOP. A nail bypasses overcurrent protection — this will cause a fire or electrocution under fault. Replace with a fuse of the correct amperage rating immediately.",
        "रुकें! नाखून से फ्यूज का काम नहीं होता — आग या करंट का खतरा है। सही एम्पीयर का फ्यूज लगाएं।",
        "TO FIX: Remove the nail. Install a fuse with the correct amperage rating.",
    ),
    "foil used as fuse": (
        "STOP. Foil bypasses overcurrent protection and will cause a fire under fault conditions. Replace with the correct rated fuse immediately.",
        "रुकें! फॉयल से फ्यूज नहीं बनता — आग लग सकती है। सही रेटिंग का फ्यूज लगाएं।",
        "TO FIX: Remove the foil. Install the correct rated fuse as marked on the fuse holder.",
    ),
    "string tied to throttle": (
        "STOP. A string tied to the throttle is a dangerous workaround — engine speed cannot be safely controlled. Do not operate. Replace with the OEM throttle linkage.",
        "रुकें! धागे से थ्रॉटल बांधना खतरनाक है। मशीन मत चलाएं। असली थ्रॉटल लिंकेज लगाएं।",
        "TO FIX: Remove the string. Install the correct OEM throttle linkage or cable.",
    ),
    "safety bypass detected": (
        "STOP. Bypassing a safety interlock removes a critical protection designed to prevent serious injury. Do not operate the machine. Restore the interlock and have it inspected before use.",
        "रुकें! सेफ्टी स्विच बायपास करना बहुत खतरनाक है। मशीन मत चलाएं। पहले सेफ्टी स्विच ठीक करवाएं।",
        "TO FIX: Restore the safety interlock. If the switch is faulty, replace it with an identical OEM safety switch.",
    ),
}


def _guard_dangerous_workaround(problem_text: str) -> Optional[dict]:
    """Check for dangerous mechanical workarounds in farmer text."""
    matched_reason = None
    for pattern, reason in _DANGEROUS_WORKAROUND_PATTERNS:
        if pattern.search(problem_text or ""):
            matched_reason = reason
            break
    if not matched_reason:
        return None

    logger.warning("🛡️ GUARD: Dangerous workaround (%s)", matched_reason)
    default = ("STOP. This is a dangerous workaround. Replace with the OEM part immediately.", "रुकें! यह खतरनाक तरीका है। तुरंत असली पुर्जा लगाएं।", "")
    warn_en, warn_hi, fix_en = _WORKAROUND_WARNINGS.get(matched_reason, default)
    warnings_en = [warn_en, fix_en] if fix_en else [warn_en]

    return {
        "technical_analysis": f"Dangerous workaround detected: {matched_reason}.",
        "safety_warnings_en": warnings_en,
        "safety_warnings_hi": [warn_hi],
    }


# ── Guard: Electric shock / flood patterns ───────────────────────────────────

_ELECTRIC_SHOCK_PATTERNS = [
    re.compile(r'\bcurrent\s+lag(?:a|i)\b', re.IGNORECASE),
    re.compile(r'\bbijli\s+lag(?:i|a)\b', re.IGNORECASE),
    re.compile(r'\belectric(?:al)?\s+shock\b', re.IGNORECASE),
    re.compile(r'\bjhatka\s+lag(?:a|i)\b', re.IGNORECASE),
    re.compile(r'\bshock\s+lag(?:a|i)\b', re.IGNORECASE),
]
_SHOCK_EXCLUSIONS = [
    re.compile(r'\btingling\b', re.IGNORECASE),
    re.compile(r'\bmild\s+shock\b', re.IGNORECASE),
    re.compile(r'\bhalka\s+current\b', re.IGNORECASE),
]
_FLOODED_MOTOR_PATTERNS = [
    re.compile(r'(?:standing|sitting|submerged|lying)\s+in\s+(?:water|flood|paani)', re.IGNORECASE),
    re.compile(r'\d+\s*inch(?:es)?\s+(?:of\s+)?(?:water|paani)', re.IGNORECASE),
]
_ELECTRIC_HAZARD_MACHINES = {"electric_motor", "submersible_pump", "water_pump"}


def _guard_electric_hazard(problem_text: str, machine_type: str) -> Optional[dict]:
    """Check for electric shock or flooded motor in farmer text."""
    prob_lower = (problem_text or "").lower()
    if any(p.search(prob_lower) for p in _SHOCK_EXCLUSIONS):
        return None
    is_shock = any(p.search(prob_lower) for p in _ELECTRIC_SHOCK_PATTERNS)
    is_flooded = (machine_type in _ELECTRIC_HAZARD_MACHINES and any(p.search(prob_lower) for p in _FLOODED_MOTOR_PATTERNS))
    if not (is_shock or is_flooded):
        return None

    logger.warning("🛡️ GUARD: Electric shock/flood [%s]", machine_type)
    if is_shock:
        return {
            "technical_analysis": "Electric shock reported. Immediate MCB cutoff and emergency medical response required.",
            "safety_warnings_en": ["EMERGENCY — electric shock. (1) Cut the main MCB immediately. (2) Call emergency services (ambulance) now. (3) Do not operate any equipment until certified safe."],
            "safety_warnings_hi": ["आपातकाल — करंट लगा है। तुरंत मेन MCB काटें और एम्बुलेंस को फोन करें।"],
        }
    else:
        return {
            "technical_analysis": "Motor submerged in water — electrocution risk if energised.",
            "safety_warnings_en": ["DANGER — do NOT start or energise this motor while it is submerged in water. Cut the main MCB immediately. Allow the motor to dry completely."],
            "safety_warnings_hi": ["खतरा — पानी में डूबी मोटर को बिल्कुल मत चलाएं। तुरंत मेन MCB काटें।"],
        }


# ── Guard: Emergency hazard patterns (fire, arc, entanglement, etc.) ─────────

from dataclasses import dataclass as _dc


class HazardLevel(IntEnum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@_dc
class HazardRule:
    category: str
    level: HazardLevel
    patterns: list
    message_en: str
    message_hi: str


_HAZARD_MESSAGES: dict = {
    "high_pressure_fluid_release": {
        "en": "STOP THE ENGINE IMMEDIATELY. High-pressure fluid can penetrate skin and cause fatal injury. Do NOT approach the spray. Cut all power. Even a pin-hole leak at pressure can inject fluid deep into tissue — seek emergency medical attention for ANY fluid injection injury, no matter how small it looks.",
        "hi": "इंजन तुरंत बंद करें। तेज दबाव वाला तरल त्वचा में घुसकर जानलेवा चोट कर सकता है। स्प्रे के पास बिल्कुल न जाएं। बिजली काटें। छोटा सा छेद भी तरल को गहराई तक पहुंचा सकता है — किसी भी तरल इंजेक्शन की चोट के लिए तुरंत डॉक्टर से संपर्क करें।",
    },
    "active_electrical_arc": {
        "en": "STOP — CUT POWER IMMEDIATELY AT THE MAIN BREAKER. Exposed live conductor with sparking detected. Do NOT touch anything. Keep all people away. Call an electrician. Do not attempt to cover or tape the wire while power is connected.",
        "hi": "तुरंत मेन स्विच से बिजली काटें। खुला तार चिंगारी छोड़ रहा है। कुछ भी न छुएं। सभी को दूर रखें। इलेक्ट्रीशियन को बुलाएं। बिजली चालू रहने पर तार को ढकने या टेप लगाने की कोशिश न करें।",
    },
    "rotating_machinery_entanglement": {
        "en": "STOP THE ENGINE IMMEDIATELY. Unguarded rotating machinery is lethal. Do NOT approach until all rotation has completely stopped. A rotating PTO shaft can entangle clothing or limbs in under 0.5 seconds — you cannot react fast enough.",
        "hi": "इंजन तुरंत बंद करें। बिना गार्ड के घूमने वाली मशीन जानलेवा है। पूरी तरह रुकने तक पास न जाएं। घूमता PTO शाफ्ट 0.5 सेकंड में कपड़े या शरीर का हिस्सा लपेट सकता है — आप रिएक्ट नहीं कर पाएंगे।",
    },
    "active_fire_or_smoke": {
        "en": "STOP THE ENGINE IMMEDIATELY. CUT ALL POWER. Smoke or flames from machinery indicate active fire risk. Use CO2 or dry powder extinguisher only — NEVER water on electrical fires. Evacuate if flames are visible. Call emergency services if fire spreads.",
        "hi": "इंजन तुरंत बंद करें। सारी बिजली काटें। मशीन से धुआं या आग निकलना सक्रिय आग का खतरा है। CO2 या ड्राई पाउडर बुझाने वाला यंत्र इस्तेमाल करें — बिजली की आग पर पानी बिल्कुल न डालें। आग दिखे तो तुरंत दूर हटें। आग फैले तो आपातकालीन सेवाओं को बुलाएं।",
    },
    "exposed_live_conductor": {
        "en": "STOP — CUT POWER IMMEDIATELY. Exposed live wire detected. Do not touch anything. Keep all people away. Call an electrician.",
        "hi": "तुरंत बिजली काटें। खुला तार दिख रहा है। कुछ न छुएं। सभी को दूर रखें। इलेक्ट्रीशियन को बुलाएं।",
    },
}

_EMERGENCY_HAZARD_RULES: list = [
    HazardRule(category="high_pressure_fluid_release", level=HazardLevel.CRITICAL, patterns=[
        re.compile(r'\b(?:oil|fuel|diesel|petrol|hydraulic|fluid|coolant|chemical|tel)\b.*?\b(?:hose|line|pipe|tube|fitting|connection)\b.*?\b(?:burst|ruptured|split|cracked|broke|spraying|spray|leaking|blown|exploded|phat\s+gaya|fat\s+gaya)\b', re.I),
        re.compile(r'\b(?:burst|ruptured|spraying|leaking)\b.*?\b(?:hydraulic|oil|fuel|diesel)\b.*?\b(?:hose|line|pipe)\b', re.I),
        re.compile(r'\b(?:spraying|leaking|shooting|pouring|burst|blew)\s+(?:oil|fuel|diesel|hydraulic\s+fluid|tel)\s+(?:from|out\s+of|everywhere)', re.I),
        re.compile(r'\b(?:hose|pipe|line|tel|oil)\s+(?:phat\s+gaya|fat\s+gaya|chhidak\s+raha|bahar\s+aa\s+raha|leak\s+kar\s+raha)', re.I),
        re.compile(r'\b(?:high\s+)?pressure\s+(?:leak|burst|spray|release|blow)', re.I),
    ], message_en=_HAZARD_MESSAGES["high_pressure_fluid_release"]["en"], message_hi=_HAZARD_MESSAGES["high_pressure_fluid_release"]["hi"]),
    HazardRule(category="active_electrical_arc", level=HazardLevel.CRITICAL, patterns=[
        re.compile(r'\b(?:exposed|bare|broken|damaged|cut|frayed|khula)\s+(?:wire|taar|cable|conductor|terminal)\s+(?:is\s+)?(?:sparking|sparks|arcing|smoking|glowing|live|energized|chingari)', re.I),
        re.compile(r'\b(?:taar|wire)\s+(?:se\s+)?(?:spark|chingari|current)\s+(?:aa\s+raha|nikal\s+raha|lag\s+raha)', re.I),
        re.compile(r'\b(?:current|bijli|shock)\s+(?:lag\s+raha|aa\s+raha)\b', re.I),
    ], message_en=_HAZARD_MESSAGES["active_electrical_arc"]["en"], message_hi=_HAZARD_MESSAGES["active_electrical_arc"]["hi"]),
    HazardRule(category="rotating_machinery_entanglement", level=HazardLevel.CRITICAL, patterns=[
        re.compile(r'\b(?:PTO|pto|shaft|belt|chain|pulley|flywheel|rotor|blade)\s+(?:is\s+)?(?:spinning|rotating|moving|running|engaged|turning|ghoom\s+raha)\s+(?:and|while|but|with)\s+(?:guard|cover|shield)\s+(?:is\s+)?(?:missing|removed|broken|off|open|nahi\s+hai)', re.I),
        re.compile(r'\b(?:PTO|pto|belt|chain)\s+(?:ghoom\s+raha|chal\s+raha)\s+(?:guard|cover)\s+(?:nahi|missing|hata\s+hua)', re.I),
        re.compile(r'\b(?:guard|cover|shield)\s+(?:is\s+)?(?:missing|removed|off)\s+(?:and|but|while)\s+(?:the\s+)?(?:shaft|PTO|belt|chain)\s+(?:is\s+)?(?:spinning|rotating|moving)', re.I),
    ], message_en=_HAZARD_MESSAGES["rotating_machinery_entanglement"]["en"], message_hi=_HAZARD_MESSAGES["rotating_machinery_entanglement"]["hi"]),
    HazardRule(category="active_fire_or_smoke", level=HazardLevel.CRITICAL, patterns=[
        re.compile(r'\b(?:smoke|dhuan|flames|aag|fire|burning)\s+(?:coming|rising|pouring|billowing|visible|detected|aa\s+raha|nikal\s+raha)\s+(?:from|out\s+of|inside|near)\s+(?:the\s+)?(?:engine|motor|alternator|generator|panel|wiring|battery|tank|machine)', re.I),
        re.compile(r'\b(?:machine|engine|motor|alternator|generator)\s+(?:se\s+)?(?:smoke|dhuan|aag|fire)\s+(?:aa\s+raha|nikal\s+raha|lagi\s+hai)', re.I),
        re.compile(r'\b(?:electrical|burning)\s+(?:smell|odor|gandh)\s+(?:from|near|around)\s+(?:engine|motor|alternator|panel)', re.I),
    ], message_en=_HAZARD_MESSAGES["active_fire_or_smoke"]["en"], message_hi=_HAZARD_MESSAGES["active_fire_or_smoke"]["hi"]),
    HazardRule(category="exposed_live_conductor", level=HazardLevel.HIGH, patterns=[
        re.compile(r'\b(?:exposed|bare|khula|nanga)\s+(?:wire|taar|cable|conductor|terminal|connection)\b', re.I),
        re.compile(r'\b(?:wire|taar)\s+(?:ki\s+)?(?:insulation|covering)\s+(?:is\s+)?(?:damaged|broken|missing|cut|removed|kharaab|nahi\s+hai)', re.I),
    ], message_en=_HAZARD_MESSAGES["exposed_live_conductor"]["en"], message_hi=_HAZARD_MESSAGES["exposed_live_conductor"]["hi"]),
]


def _guard_emergency_hazard(problem_text: str) -> Optional[dict]:
    """Check for emergency hazards (fire, arc, entanglement, high-pressure, exposed conductor)."""
    if not problem_text:
        return None

    matches: list = []
    for rule in _EMERGENCY_HAZARD_RULES:
        for pattern in rule.patterns:
            if pattern.search(problem_text):
                matches.append(rule)
                break

    if not matches:
        return None

    best = max(matches, key=lambda r: r.level.value)
    logger.warning("🚨 EMERGENCY HAZARD: category=%s level=%s query='%s...'", best.category, best.level.name, problem_text[:60])

    return {
        "technical_analysis": best.message_en,
        "safety_warnings_en": [best.message_en],
        "safety_warnings_hi": [best.message_hi],
    }


# ── Public API ────────────────────────────────────────────────────────────────

# Mapping from guard function → severity level (used for priority resolution)
_GUARD_SEVERITY = {
    "emergency_hazard": GuardSeverity.CRITICAL,
    "electric_hazard": GuardSeverity.HIGH,
    "dangerous_workaround": GuardSeverity.HIGH,
}


def run_text_hazard_checks(
    text: str,
    machine_type: str,
) -> Optional[GuardResult]:
    """
    Run all deterministic text-hazard guards against a single text string.

    Called by /verify_step, /locate_part, /agent/next, and /diagnose before
    any LLM call. Returns GuardResult if blocked, None if safe.

    Fails closed: if any guard crashes, returns a blocking GuardResult.
    """
    if not text or not text.strip():
        return None

    for guard_fn, guard_name in [
        (_guard_emergency_hazard, "emergency_hazard"),
        (lambda t: _guard_electric_hazard(t, machine_type), "electric_hazard"),
        (_guard_dangerous_workaround, "dangerous_workaround"),
    ]:
        try:
            result = guard_fn(text)
        except Exception as exc:
            logger.exception("Safety guard '%s' crashed — failing closed", guard_name)
            return GuardResult(
                blocked=True,
                severity=GuardSeverity.HIGH,
                technical_analysis="Safety validation failed. Unable to verify scene safety.",
                safety_warnings_en=["Safety validation failed. Stop operation and contact a certified mechanic."],
                safety_warnings_hi=["सुरक्षा जाँच विफल रही। मशीन बंद रखें और प्रमाणित मैकेनिक से संपर्क करें।"],
                guard_name="guard_failure",
            )

        if result is not None and isinstance(result, dict):
            return GuardResult(
                blocked=True,
                severity=_GUARD_SEVERITY.get(guard_name, GuardSeverity.HIGH),
                technical_analysis=result.get("technical_analysis", ""),
                safety_warnings_en=result.get("safety_warnings_en", []),
                safety_warnings_hi=result.get("safety_warnings_hi", []),
                guard_name=guard_name,
            )

    return None


def check_multiple_texts(
    texts: dict[str, str],
    machine_type: str,
) -> Optional[GuardResult]:
    """
    Check multiple text fields, returning the highest-severity hazard found.
    Runs ALL checks, not just the first — so all hazards are logged.
    """
    all_results: list[GuardResult] = []
    for field_name, text in texts.items():
        if not text:
            continue
        result = run_text_hazard_checks(text, machine_type)
        if result is not None:
            all_results.append(result)

    if not all_results:
        return None

    # Return the highest-severity result
    return max(all_results, key=lambda r: r.severity.value)