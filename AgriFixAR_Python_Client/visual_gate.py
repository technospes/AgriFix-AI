from __future__ import annotations
import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from PIL import Image
import io

logger = logging.getLogger(__name__)

_VISION_MODEL = "gemini-2.5-flash-lite"
VISION_CONFIDENCE_THRESHOLD = 0.65

# ── Visual Part Normalization Map ──
_CANONICAL_MAP = {
    "starter relay": "relay",
    "contactor": "relay",
    "relay": "relay",
    "starter_relay": "relay"
}

_VERIFICATION_PROMPT = """\
You are a precision diagnostic verification agent for farm machinery repair.
Your ONLY job: confirm whether the part described below is visible in this camera image,
and whether it shows the fault described.

DATABASE CHUNK (what the repair manual says about this fault):
{chunk_text}

TARGET PART(S) TO FIND: {target_parts}

STRICT INSTRUCTIONS:
1. Look for the target part(s) in the image.
2. Answer in ONLY this JSON format — no markdown, no explanation:
{{
  "part_visible": true | false,
  "part_id": "<snake_case_id of the most relevant part found, or 'unknown'>",
  "fault_visible": true | false,
  "fault_description": "<ONE sentence — what you actually see, or 'Part not visible'>",
  "confidence": 0.0-1.0,
  "gate_verdict": "PASS" | "FAIL" | "UNCLEAR"
}}

gate_verdict rules:
  PASS   → part is clearly visible AND fault matches the database description
  FAIL   → part is visible but fault does NOT match (wrong part or different issue)
  UNCLEAR → part is not clearly visible, image is blurry, or insufficient information
"""

CAMERA_PROMPT_EN = "Please aim your camera at the {part_area} of your machine. I need to visually confirm the issue before showing you repair steps."
CAMERA_PROMPT_HI = "कृपया अपना कैमरा मशीन के {part_area} पर लगाएं। मरम्मत के चरण दिखाने से पहले मुझे दृष्टि से समस्या की पुष्टि करनी होगी।"

_PART_AREA_LABELS: Dict[str, str] = {
    "fuel_filter": "fuel filter area", "battery_terminal": "battery area",
    "air_filter": "air filter", "coolant_reservoir": "coolant reservoir",
    "motor_winding": "electric motor", "impeller": "pump impeller area",
    "foot_valve": "foot valve", "capacitor": "capacitor", "relay": "relay",
    "pto_shaft": "PTO shaft", "shear_bolt": "shear bolt",
    "primer_pump": "primer pump", "injector": "fuel injector area",
    "glow_plug": "glow plug", "drive_belt": "drive belt", "alternator": "alternator"
}

@dataclass
class GateResult:
    gate_passed: bool
    verdict: str
    part_id: str
    fault_visible: bool
    fault_description: str
    confidence: float
    target_parts: List[str]
    machine_type: str
    error: Optional[str] = None

    def re_examine_response(self) -> Dict[str, Any]:
        part_label = _PART_AREA_LABELS.get(self.part_id or (self.target_parts[0] if self.target_parts else ""), "relevant area")
        if self.verdict == "UNCLEAR":
            reason_en = f"I can see your camera image, but I cannot clearly confirm the {part_label}. Please move closer."
            reason_hi = f"मैं कैमरे की छवि देख सकता हूं, लेकिन {part_label} स्पष्ट नहीं है। कृपया करीब जाएं।"
        elif self.verdict == "FAIL":
            reason_en = f"I'm looking at the camera feed — the {part_label} does not match the issue. What I see: {self.fault_description}"
            reason_hi = f"मैं कैमरे की छवि देख रहा हूं — {part_label} डेटाबेस में मिली समस्या से मेल नहीं खाती।"
        else:
            reason_en = "I could not confirm the issue visually. Please re-aim the camera."
            reason_hi = "मैं दृष्टि से समस्या की पुष्टि नहीं कर सका। कृपया कैमरा फिर से लगाएं।"

        return {
            "status": "visual_gate_failed", "gate_passed": False,
            "verdict": self.verdict, "message_en": reason_en, "message_hi": reason_hi,
            "camera_prompt_en": CAMERA_PROMPT_EN.format(part_area=part_label),
            "camera_prompt_hi": CAMERA_PROMPT_HI.format(part_area=part_label),
            "visual_observation": self.fault_description, "confidence": round(self.confidence, 3),
            "steps": [],
        }

    def to_session_observation(self) -> Dict[str, str]:
        if self.gate_passed and self.part_id and self.part_id != "unknown":
            return {self.part_id: self.fault_description}
        return {}


# Stopwords that should never be treated as part names
_PART_STOPWORDS: frozenset = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor", "so",
    "yet", "both", "either", "each", "more", "most", "other", "some",
    "such", "than", "that", "this", "these", "those", "it", "its",
    "my", "your", "his", "her", "our", "their", "i", "we", "you",
    "he", "she", "they", "what", "which", "who", "when", "where",
    "how", "if", "as", "up", "out", "about", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "there", "here", "then", "any", "all", "and", "or", "but",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "also", "very", "just", "now", "only", "too", "really", "still",
    "may", "needed", "need", "must", "can", "could", "should",
    "will", "would", "shall", "might", "has", "had", "having",
    "does", "did", "doing", "been", "being", "get", "got", "gotten",
    "put", "set", "see", "saw", "seen", "use", "used", "using",
    "one", "two", "three", "first", "second", "last", "next",
    "new", "old", "good", "bad", "high", "low", "big", "small",
})

# Valid part ID pattern: snake_case with at least one underscore OR
# a known compound word from the registry
_VALID_PART_PATTERN = re.compile(
    r'\b(?:[a-z]+_[a-z][a-z0-9_]+)\b'  # snake_case with underscore
)

def extract_target_parts(rag_context: str, max_parts: int = 3) -> List[str]:
    """
    Extract target part IDs from RAG context.
    
    Only extracts valid snake_case identifiers (containing underscores)
    or JSON field values. Filters out English stopwords that regex
    incorrectly captures.
    """
    found: List[str] = []
    
    # Method 1: Extract from JSON fields (most reliable)
    cue_match = re.findall(r'"(?:visual_cue|required_part)"\s*:\s*"([^"]+)"', rag_context)
    found.extend(cue_match)
    
    # Method 2: Extract snake_case part IDs from structured text
    parts_match = re.findall(r'PARTS?[:\s]+([^\n]+)', rag_context, re.IGNORECASE)
    for match in parts_match:
        # Only extract valid snake_case identifiers
        valid_parts = _VALID_PART_PATTERN.findall(match.lower())
        found.extend(valid_parts)

    # Dedup and filter: remove stopwords, invalid entries
    seen: set = set()
    unique: List[str] = []
    for p in found:
        p = p.strip().lower()
        # Skip stopwords
        if p in _PART_STOPWORDS:
            continue
        # Skip invalid entries
        if p in ("unknown", "none", "null", ""):
            continue
        # Skip single-character or very short tokens
        if len(p) < 3:
            continue
        # Skip tokens without underscores (likely noise unless from JSON)
        if "_" not in p and p not in {"relay", "capacitor"}:  # known single-word parts
            continue
        if p not in seen:
            seen.add(p)
            unique.append(p)
    
    logger.debug("extract_target_parts: %d found → %d valid: %s",
                 len(found), len(unique), unique[:max_parts])
    return unique[:max_parts]


async def _call_gemini_vision(frame_bytes: bytes, prompt: str) -> str:
    img = Image.open(io.BytesIO(frame_bytes))
    max_px = 512
    if max(img.width, img.height) > max_px:
        scale = max_px / max(img.width, img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)

    model = genai.GenerativeModel(_VISION_MODEL)
    response = await asyncio.to_thread(
        lambda: model.generate_content(
            [{"mime_type": "image/jpeg", "data": buf.getvalue()}, prompt],
            generation_config={"temperature": 0.1, "max_output_tokens": 300},
            request_options={"timeout": 5},
        )
    )
    return response.text


def _parse_vision_response(raw: str) -> Dict[str, Any]:
    text = re.sub(r"^```json\s*", "", raw.strip())
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r",\s*}", "}", text)
    import json
    return json.loads(text)


async def run_visual_gate(
    frame_bytes: Optional[bytes],
    target_parts: List[str],
    rag_chunk_text: str,
    machine_type: str,
    confidence_threshold: float = VISION_CONFIDENCE_THRESHOLD,
) -> GateResult:
    if not frame_bytes:
        return GateResult(True, "SKIP", "unknown", False, "No camera image.", 0.0, target_parts, machine_type)
    if not target_parts:
        return GateResult(True, "SKIP", "unknown", False, "No target parts.", 0.0, [], machine_type)

    prompt = _VERIFICATION_PROMPT.format(chunk_text=rag_chunk_text[:600], target_parts=", ".join(target_parts))

    try:
        raw_response = await _call_gemini_vision(frame_bytes, prompt)
        data = _parse_vision_response(raw_response)
    except Exception as exc:
        logger.error("Visual gate error: %s", exc)
        return GateResult(True, "ERROR", "unknown", False, "Verification unavailable.", 0.0, target_parts, machine_type, str(exc))

    verdict     = str(data.get("gate_verdict", "UNCLEAR")).upper()
    part_id     = str(data.get("part_id", "unknown")).lower().strip()
    confidence  = float(data.get("confidence", 0.0))
    fault_vis   = bool(data.get("fault_visible", False))
    fault_desc  = str(data.get("fault_description", "No observation available.")).strip()

    # ── Normalize & Validate Match ──
    norm_detected = _CANONICAL_MAP.get(part_id, part_id)
    norm_targets = [_CANONICAL_MAP.get(tp.lower(), tp.lower()) for tp in target_parts]

    gate_passed = False
    if verdict == "PASS" and confidence >= confidence_threshold and fault_vis:
        if norm_detected in norm_targets or not target_parts or norm_detected == "unknown":
            gate_passed = True
        else:
            logger.warning(f"❌ Visual Mismatch: Targets {norm_targets} != Detected {norm_detected}")
            verdict = "FAIL"
            fault_desc = f"Mismatch: Expected {', '.join(norm_targets)}, but saw {norm_detected}. {fault_desc}"

    logger.info("Visual gate [%s]: verdict=%s part=%s conf=%.2f passed=%s", machine_type, verdict, norm_detected, confidence, gate_passed)

    return GateResult(
        gate_passed=gate_passed, verdict=verdict, part_id=norm_detected,
        fault_visible=fault_vis, fault_description=fault_desc,
        confidence=confidence, target_parts=target_parts, machine_type=machine_type,
    )

def get_camera_prompt(target_parts: List[str], language: str = "en") -> str:
    area = _PART_AREA_LABELS.get(target_parts[0], target_parts[0].replace("_", " ")) if target_parts else "the main machine body"
    return CAMERA_PROMPT_HI.format(part_area=area) if language == "hi" else CAMERA_PROMPT_EN.format(part_area=area)