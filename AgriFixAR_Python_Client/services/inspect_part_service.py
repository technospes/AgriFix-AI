from __future__ import annotations
import logging
from typing import Optional
from utils.vision_client import vision_call

logger = logging.getLogger(__name__)

INSPECT_PROMPT = """You are inspecting ONE verified part on farm machinery.
Target: {required_part} | Area: {area_hint} | Machine: {machine_label}

OUTCOME (pick one):
  healthy      — no damage. Normal wear is NOT damage.
  damaged      — cracks, fraying, bulging, corrosion, burns, leaks, breaks,
                 bends, missing pieces, loose connections, melted plastic,
                 exposed wires, rust.
  unclear      — hidden, dirty, dark, cannot assess.
  wrong_target — image shows a DIFFERENT part than the target.

CONFIDENCE (0.0-1.0). Calibrate honestly:
  0.9+ = certain, well-lit, clear view.
  0.6-0.8 = minor obstruction or odd angle.
  0.3-0.5 = partial view or ambiguous.
  <0.3 = mostly guessing.

REPAIRABILITY (only if damaged):
  field_repairable  — basic tools: tighten, clean, patch, replace consumable.
  mechanic_required — special tools/skill: welding, rewiring, precision fit.
  replace_only      — destroyed or unsafe to repair.
  unknown           — path unclear from this image.

Return ONLY JSON:
{{
  "outcome": "healthy"|"damaged"|"unclear"|"wrong_target",
  "confidence": 0.0-1.0,
  "severity": "none"|"minor"|"moderate"|"severe",
  "damage_description": "<farmer-friendly, empty if healthy>",
  "damage_description_hi": "<same in simple Hindi, empty if healthy>",
  "repairability": "field_repairable"|"mechanic_required"|"replace_only"|"unknown",
  "observations": ["<short factual note>"],
  "safety_concern": true|false,
  "safety_note": "<only if safety_concern=true, else empty>"
}}"""

def _parse_confidence(value) -> float:
    """Safely cast Gemini's confidence to float in [0.0, 1.0]. Falls back to 0.5."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, c))

async def inspect_part_service(
    image_bytes: bytes,
    machine_type: str,
    required_part: str,
    area_hint: str = "engine_compartment",
    language: str = "en",
) -> dict:
    """Inspect a verified part for damage. Returns InspectionResult."""
    
    from utils.machine_registry import get_profile_or_default
    profile = get_profile_or_default(machine_type)
    
    prompt = INSPECT_PROMPT.format(
        required_part=required_part.replace("_", " "),
        area_hint=area_hint.replace("_", " "),
        machine_label=profile.label_en,
    )
    
    try:
        response = await vision_call(
            image_bytes=image_bytes,
            prompt=prompt,
            temperature=0.2,
        )
        
        from utils.json_repair import repair_json
        data = repair_json(response)
        
        return {
            "outcome": data.get("outcome", "unclear"),
            "severity": data.get("severity", "none"),
            "damage_description": data.get("damage_description", ""),
            "damage_description_hi": data.get("damage_description_hi", ""),
            "repairability": data.get("repairability", "unknown"),
            "observations": data.get("observations", []),
            "safety_concern": data.get("safety_concern", False),
            "safety_note": data.get("safety_note", ""),
            "confidence": _parse_confidence(data.get("confidence", 0.5)),
        }
        
    except Exception as exc:
        logger.error("Inspect part failed: %s", exc)
        return {
            "outcome": "unclear",
            "severity": "none",
            "damage_description": "",
            "damage_description_hi": "",
            "repairability": "unknown",
            "observations": [],
            "safety_concern": False,
            "safety_note": "",
            "confidence": 0.0,
        }