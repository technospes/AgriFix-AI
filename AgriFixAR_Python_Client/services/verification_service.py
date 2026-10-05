from __future__ import annotations
import asyncio
import os
import json
import logging
from utils.json_repair import repair_json
from utils.image_utils import resize_image
from utils.vision_client import vision_call
from utils.machine_registry import (
    get_area_farmer_description,
    get_allowed_area_ids,
    is_electric_machine,
    get_profile,
    get_compact_parts_list,
)
VERIFY_MAX_IMAGE_DIM = int(os.getenv("VERIFY_MAX_IMAGE_DIM", "720"))
logger = logging.getLogger(__name__)
CONFIDENCE_THRESHOLD_PASS = 0.70

# ── Jargon substitution map ───────────────────────────────────────────────────
_UNIVERSAL_JARGON_MAP = (
    'corrosion→"white powder on metal"; '
    'insulation_damage→"wire looks fuzzy/split"; '
    'belt_tension→"belt feels loose/stretched"; '
    'cavitation→"pump rattling"; '
    'MCB_fault→"switch popped up in panel".'
)

# ── Per-machine observation focus hints ──────────────────────────────────────
_MACHINE_OBS_HINT = {
    "tractor":          "Look for loose/corroded cables, belt condition, oil/coolant leaks.",
    "harvester":        "Look for crop blockages in visible openings, belt/chain wear, chaff on radiator.",
    "thresher":         "Look for crop jam at feed inlet or concave, belt slip, bearing damage.",
    "submersible_pump": "Look for tripped MCB (switch up=tripped), bulging capacitor, wiring burns.",
    "water_pump":       "Look for air leaks at pipe joints, priming plug open, seal water leak at shaft.",
    "electric_motor":   "Look for tripped relay (red button up), bulging capacitor can, burnt terminal smell.",
    "power_tiller":     "Look for fuel level, air filter clog, clutch cable tension, tine bolt presence.",
    "chaff_cutter":     "FIRST check safety guard is in place. Then look for blade sharpness and belt.",
    "diesel_engine":    "Look for exhaust smoke colour (black/blue/white), oil level, air filter clog.",
    "rotavator":        "Look for missing/bent blades, shear bolt intact, gearbox oil leak.",
    "generator":        "Look for breaker position (tripped=up), capacitor bulge, AVR board condition.",
}

# ── Machine-part compatibility map ───────────────────────────────────────────
# Parts that are IMPOSSIBLE to appear on a given machine.
# If Gemini detects such a part, the verification is flagged as a mismatch.
# Format: machine_type → set of part keywords that do NOT belong to it.
_MACHINE_INCOMPATIBLE_PARTS: dict[str, set[str]] = {
    "submersible_pump": {"pto", "flywheel", "threshing_cylinder", "chaff", "combine",
                         "blade", "crop", "harvesting", "tractor_engine"},
    "water_pump":       {"capacitor", "winding", "mcb", "relay", "motor_terminal",
                         "pto", "threshing_cylinder"},
    "electric_motor":   {"fuel", "injector", "carburettor", "fuel_filter",
                         "pto_shaft", "threshing_cylinder"},
    "tractor":          {"capacitor_can", "motor_winding", "mcb_panel",
                         "threshing_cylinder", "chaff_blower"},
    "chaff_cutter":     {"impeller", "suction_pipe", "foot_valve",
                         "mcb", "capacitor", "motor_winding"},
    "rotavator":        {"capacitor", "winding", "suction_pipe", "mcb"},
}

# ── Diagnosis-to-part consistency map ─────────────────────────────────────────
# If a step diagnosis mentions component A but camera finds component B,
# and they're from different subsystems, flag mismatch.
# Maps a required_part keyword → the subsystem it belongs to.
_PART_SUBSYSTEM: dict[str, str] = {
    "capacitor":          "electrical",
    "motor_winding":      "electrical",
    "mcb":                "electrical",
    "relay":              "electrical",
    "battery_terminal":   "electrical",
    "alternator":         "electrical",
    "starter_motor":      "electrical",
    "fuel_filter":        "fuel",
    "injector":           "fuel",
    "fuel_pump":          "fuel",
    "primer_pump":        "fuel",
    "carburettor":        "fuel",
    "impeller":           "hydraulic",
    "foot_valve":         "hydraulic",
    "suction_pipe":       "hydraulic",
    "pressure_pipe":      "hydraulic",
    "hydraulic_cylinder": "hydraulic",
    "pto_shaft":          "mechanical",
    "drive_belt":         "mechanical",
    "shear_bolt":         "mechanical",
    "blade":              "mechanical",
    "flywheel":           "mechanical",
    "air_filter":         "air",
    "radiator":           "cooling",
    "coolant_reservoir":  "cooling",
    "thermostat":         "cooling",
}


def _get_subsystem(part_id: str) -> str | None:
    """Return subsystem for a part_id, or None if unknown."""
    key = part_id.lower().replace(" ", "_")
    for token, sub in _PART_SUBSYSTEM.items():
        if token in key:
            return sub
    return None


async def verify_step_with_gemini(
    image_bytes: bytes,
    step_text: str,
    required_part: str | None,
    area_hint: str,
    machine_type: str,
    problem_context: str,
    attempt_count: int,
    language: str = "en",
    include_hindi: bool = False,
    previous_steps: str = "[]",
    tracking_scope: str = "component",
) -> dict:
    """Verify a repair step for any supported farm machine — token-optimised."""
    logger.info(
        f"🔍 Vision: machine={machine_type} part={required_part} "
        f"area={area_hint} attempt={attempt_count}"
    )

    try:
        resized_bytes = await resize_image(image_bytes, max_dim=VERIFY_MAX_IMAGE_DIM)

        area_desc   = get_area_farmer_description(machine_type, area_hint, language)
        req_part_str = required_part or ""
        is_electric = is_electric_machine(machine_type)
        obs_hint    = _MACHINE_OBS_HINT.get(
            getattr(get_profile(machine_type), "machine_id", machine_type),
            "Look for visible damage, loose parts, leaks, or blockages."
        )
        electric_flag = "⚡ UNSAFE if live wires/terminals visible near hands.\n" if is_electric else ""
        lang_note     = "feedback in Hindi preferred.\n" if language == "hi" else ""

        # ── Visual memory ──────────────────────────────────────────────────
        history_block = ""
        try:
            prev = json.loads(previous_steps) if previous_steps else []
            if not isinstance(prev, list):
                prev = []
            passed = [s for s in prev if s.get("status") in ("answered", "pass", "verified")]
            if passed:
                lines = [
                    f"  • {s.get('detected_part', '?')}: {s.get('feedback', s.get('status', '?'))}"
                    for s in passed[-5:]
                ]
                history_block = (
                    "\nALREADY CONFIRMED THIS SESSION (do not re-describe these parts):\n"
                    + "\n".join(lines) + "\n"
                )
        except Exception:
            pass

        if tracking_scope == "assembly":
            identity_check = f"""
# IDENTITY CHECK — ASSEMBLY MODE:
# Confirm the camera is pointing directly at the {area_hint} ({area_desc}) of the machine.
Find: {area_hint} structural area | attempt={attempt_count}
"""
        else:
            identity_check = f"""
# IDENTITY CHECK — COMPONENT MODE:
# Confirm the visible component is EXACTLY: "{req_part_str}" located in "{area_hint}" ({area_desc})
Find: {req_part_str} in {area_hint} ({area_desc}) | attempt={attempt_count}
"""

        prompt = f"""Farm machinery camera verification. Farmer tapped Analyze on their {machine_type}.
{electric_flag}{lang_note}{history_block}
Context: problem={problem_context} | step="{step_text}"

ACTION CHECK:
The farmer is trying to perform the action described in the step.
If the correct part is visible but the action has NOT been performed
(for example pedal not pressed, lever not moved, cable still connected),
explain the mistake in feedback and tell the farmer what action to perform.
Do NOT say "image unclear" if the part is visible but the action is missing.
If the correct part is visible but the action is incorrect or incomplete,
status MUST be "fail".

{identity_check}
Only report a part if it is clearly visible in the image. Do not guess.

# DIRTY / OBSCURED COMPONENTS:
# If the part appears heavily dusty, muddy, or corroded but is identifiable,
# set status="pass" and note the condition in ai_observation.
# If obscured to the point of not being assessable → status="unclear",
# feedback line 2 = "Clean the part slightly and retry."

Focus: {obs_hint}
Jargon→plain: {_UNIVERSAL_JARGON_MAP}

AI observes only — never ask farmer to describe, speak, or report anything.

ai_observation: 1 sentence — what the camera sees right now in plain physical words. No jargon.

feedback: EXACTLY two short lines separated by a newline.
Line 1: what the farmer did wrong.
Line 2: the exact action the farmer must perform next.

Return ONLY this JSON:
{{
  "status": "pass" | "fail" | "unclear" | "unsafe",
  "confidence": 0.0-1.0,
  "detected_part": "<physical description of what is visible>",
  "correct_part": "{required_part}",
  "ai_observation": "<1 sentence: what camera sees in plain words>",
  "feedback": "<2 lines max: line 1 = what farmer did wrong, line 2 = exact fix>",
  "feedback_hi": "<same 2 lines — simple village Hindi>",
  "safety_note": null
}}
pass=part_visible+assessable(conf≥0.70); fail=wrong_area; unclear=bad_image; unsafe=danger_visible."""

        response_text = await vision_call(
            prompt=prompt,
            image_bytes=resized_bytes,
            max_tokens=400,
            temperature=0.1,
        )

        result    = repair_json(response_text)
        raw_status = result.get("status", "unclear")
        raw_conf   = float(result.get("confidence", 0.0))

        # ── GATE 1: Machine-type compatibility check ───────────────────────
        # If Gemini detected a part that cannot exist on this machine type,
        # the image is probably misdirected (wrong machine or area).
        detected_raw = (result.get("detected_part") or "").lower()
        incompatible = _MACHINE_INCOMPATIBLE_PARTS.get(machine_type, set())
        machine_mismatch = any(token in detected_raw for token in incompatible)

        if machine_mismatch and raw_status in ("pass", "fail") and tracking_scope == "component":
            raw_status = "fail"
            raw_conf   = 0.0
            ai_feedback = result.get("feedback", "")
            result["feedback"] = ai_feedback if ai_feedback else (
                f"Wrong machine or area — detected component does not belong to a {machine_type}.\n"
                f"Aim camera at the {area_hint.replace('_', ' ')} of your {machine_type}."
            )

            ai_feedback_hi = result.get("feedback_hi", "")
            result["feedback_hi"] = ai_feedback_hi if ai_feedback_hi else (
                f"गलत मशीन या क्षेत्र — यह हिस्सा {machine_type} का नहीं है।\n"
                f"कैमरा अपने {machine_type} के {area_hint.replace('_', ' ')} पर लगाएं।"
            )

            logger.warning(
                f"verify_step: MACHINE_MISMATCH [{machine_type}] detected='{detected_raw[:60]}'"
            )

        # ── GATE 2 & 3: Component validation ────────────────────────────────
        # Only meaningful in component tracking mode with a named part —
        # assembly-mode steps have no single component to mismatch against.
        if tracking_scope == "component" and raw_status == "pass" and req_part_str:

            # GATE 2: Component vs diagnosis subsystem mismatch
            # If the detected part is from a different subsystem than required_part,
            # the farmer is looking at the wrong component entirely.
            required_sub = _get_subsystem(req_part_str)
            # Extract subsystem of whatever Gemini detected
            detected_sub = None
            for token, sub in _PART_SUBSYSTEM.items():
                if token in detected_raw:
                    detected_sub = sub
                    break
            if (required_sub and detected_sub
                    and required_sub != detected_sub):
                raw_status = "fail"
                raw_conf   = min(raw_conf, 0.45)
                
                # Keep Gemini's feedback, just append the subsystem warning
                ai_feedback = result.get("feedback", f"This is a {detected_sub} component.")
                result["feedback"] = (
                    f"{ai_feedback}\n"
                    f"Point camera at the {area_hint.replace('_', ' ')}."
                )
                ai_feedback_hi = result.get("feedback_hi", f"यह एक {detected_sub} हिस्सा है।")
                result["feedback_hi"] = (
                    f"{ai_feedback_hi}\n"
                    f"कैमरा {area_hint.replace('_', ' ')} की तरफ करें।"
                )
                logger.warning(
                    f"verify_step: SUBSYSTEM_MISMATCH [{machine_type}] "
                    f"required={required_sub} detected={detected_sub}"
                )

            # GATE 3: Wrong-part guard (name-level)
            required_key   = req_part_str.lower().replace("_", " ")
            required_words = [w for w in required_key.split() if len(w) > 3]
            part_words_match = any(w in detected_raw for w in required_words)
            if raw_status == "pass" and not part_words_match and required_words:
                raw_status = "fail"
                raw_conf   = min(raw_conf, 0.50)
                
                # Keep Gemini's smart spatial feedback!
                ai_feedback = result.get("feedback", f"Wrong component shown.")
                result["feedback"] = (
                    f"{ai_feedback}\n"
                    f"Point camera at {area_hint.replace('_', ' ')}."
                )
                ai_feedback_hi = result.get("feedback_hi", f"गलत हिस्सा दिखाया।")
                result["feedback_hi"] = (
                    f"{ai_feedback_hi}\n"
                    f"{area_hint.replace('_', ' ')} की तरफ कैमरा करें।"
                )
                logger.info(f"verify_step: WRONG_PART [{machine_type}] "
                            f"detected='{detected_raw[:50]}' required='{required_key}'")

        # ── GATE 4: Confidence threshold enforcement ───────────────────────
        # A "pass" with confidence below CONFIDENCE_THRESHOLD_PASS is returned
        # as "need_verification" rather than silently treating it as verified.
        # This prevents low-signal confirmations from advancing the repair flow.
        display_target = area_hint.replace("_", " ") if tracking_scope == "assembly" else req_part_str.replace("_", " ")

        if raw_status == "pass" and raw_conf < CONFIDENCE_THRESHOLD_PASS:
            result["status"]   = "need_verification"
            result["verified"] = False
            result["feedback"] = (
                f"Camera cannot verify the {display_target} with sufficient confidence "
                f"(score: {raw_conf:.0%}).\n"
                "Move closer, improve lighting, hold steady, then tap Analyze."
            )
            result["feedback_hi"] = (
                f"कैमरा {display_target} की पुष्टि करने में असमर्थ (स्कोर: {raw_conf:.0%})।\n"
                "पास जाएं, अच्छी रोशनी करें, स्थिर रखें, फिर विश्लेषण दबाएं।"
            )
            result["confidence"]   = raw_conf
            result["attempt_count"] = attempt_count
            result["machine_type"]  = machine_type
            logger.info(
                f"verify_step: LOW_CONFIDENCE [{machine_type}] "
                f"part={required_part} conf={raw_conf:.2f} → need_verification"
            )
            return result

        is_verified = raw_status == "pass" and raw_conf >= CONFIDENCE_THRESHOLD_PASS
        result["verified"]      = is_verified
        result["status"]        = "verified" if is_verified else raw_status
        result["attempt_count"] = attempt_count
        result["machine_type"]  = machine_type

        if result.get("status") in ("fail", "unclear") and attempt_count >= 3:
            q = f"{machine_type} {required_part} repair location".replace(" ", "+")
            result["help_url"] = f"https://www.youtube.com/results?search_query={q}"
            result["feedback"] = (result.get("feedback", "") +
                                  " Watch a video guide (tap help button).").strip()

        logger.info(
            f"✅ Vision: {result.get('status')} conf={result.get('confidence', 0):.2f} "
            f"[{machine_type}]"
        )
        return result

    except json.JSONDecodeError as exc:
        logger.error(f"❌ Vision JSON error [{machine_type}]: {exc}")
    except Exception as exc:
        logger.error(f"❌ Vision error [{machine_type}]: {exc}")

    return _fallback_verification(required_part, machine_type, attempt_count)

def _fallback_verification(required_part: str, machine_type: str, attempt_count: int) -> dict:
    return {
        "status": "unclear", "verified": False, "confidence": 0.0,
        "detected_part": "Could not identify — image unclear",
        "correct_part": required_part, "machine_type": machine_type,
        "ai_observation": "The image is too dark or blurry to see the part.",
        "feedback": "The camera cannot see the part clearly.\nMove closer — forearm-length away — hold still, then tap Analyze.",
        "feedback_hi": "कैमरे को हिस्से के पास लाएं — हाथ की लंबाई जितनी दूरी।\nस्थिर रखें, फिर विश्लेषण दबाएं।",
        "attempt_count": attempt_count,
    }