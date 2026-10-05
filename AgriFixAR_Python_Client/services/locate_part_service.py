from __future__ import annotations
import asyncio
import os
import json
import logging
from typing import Optional
import time as _time
from utils.json_repair import repair_json
from utils.image_utils import resize_image
from utils.vision_client import vision_call
from utils.machine_registry import get_area_farmer_description, get_profile
_BBOX_CACHE_TTL_S: float = 0.8
_bbox_cache: dict = {}   # key → {result, ts, hits}
LOCATE_PART_MAX_IMAGE_DIM = int(os.getenv("LOCATE_PART_MAX_IMAGE_DIM", "512"))

logger = logging.getLogger(__name__)

_CONF_THRESHOLD  = 0.82

_AREA_DIRECTIONS: dict[str, str] = {
    "suction_side":     "the inlet/suction side of the pump",
    "discharge_side":   "the outlet/discharge side of the pump",
    "pump_body":        "the main body of the pump",
    "coupling_area":    "the coupling/shaft between motor and pump",
    "engine_bay":       "the engine compartment",
    "transmission":     "the gearbox/transmission area",
    "fuel_system":      "the fuel tank and filter area",
    "electrical_panel": "the electrical panel or control box",
    "top_cover":        "the top cover of the machine",
    "front_panel":      "the front face of the machine",
    "rear_panel":       "the rear/back of the machine",
    "left_side":        "the left side of the machine",
    "right_side":       "the right side of the machine",
    "underside":        "the bottom of the machine",
}


async def locate_part_with_gemini(
    image_bytes: bytes,
    required_part: str,
    area_hint: str,
    machine_type: str,
    attempt_count: int = 1,
    language: str = "en",
    frame_id: int = 0,
    search_roi: tuple[float, float, float] | None = None,
) -> dict:
    """
    search_roi: (cx, cy, margin) — normalised coords of last known bbox centre.
    When provided, tells Gemini to focus its search in that region.
    This reduces false positives and speeds up reasoning on cluttered frames.
    Margin is half-width of the search window (e.g. 0.30 = search ±30% of image).
    """
    """
    Ask Gemini Vision to locate a specific part and return its bounding box.

    Gemini always identifies whatever it actually sees (visible_component) and,
    when that isn't the target, reasons about the target's position relative to
    it — so camera_guidance is populated from real per-frame spatial reasoning
    rather than a generic string, even in the found=False case.

    Returns a dict:
      found=True:
        {
          "found": true,
          "bbox": [cx, cy, w, h],   # normalised 0.0–1.0, cx/cy = centre
          "confidence": 0.85,
          "part_description": "round red plug on upper-left of pump body",
          "visible_component": "priming plug",
          "camera_guidance": ""
        }

      found=False:
        {
          "found": false,
          "bbox": null,
          "confidence": 0.0,
          "part_description": null,
          "visible_component": "drain plug",
          "camera_guidance": "This is the drain plug. The priming plug is above it."
        }
    """
    # ── Server-side bbox cache check ─────────────────────────────────────────
    _cache_key = f"{machine_type}|{required_part}|{area_hint}"
    _now_ts    = _time.monotonic()
    _cached    = _bbox_cache.get(_cache_key)
    if _cached and (_now_ts - _cached["ts"]) < _BBOX_CACHE_TTL_S:
        _cached["hits"] += 1
        _cached["result"]["frame_id"] = frame_id  # echo caller's frame_id
        logger.info(
            f"⚡ locate_part CACHE HIT key={_cache_key} "
            f"age={_now_ts - _cached['ts']:.2f}s hits={_cached['hits']} "
            f"conf={_cached['result']['confidence']} — skipped Gemini"
        )
        return _cached["result"]

    logger.info(
        f"🎯 locate_part: machine={machine_type} part={required_part} "
        f"area={area_hint} attempt={attempt_count}"
    )

    try:
        resized_bytes = await resize_image(image_bytes, max_dim=LOCATE_PART_MAX_IMAGE_DIM, quality=82)

        area_desc      = get_area_farmer_description(machine_type, area_hint, language)
        area_direction = _AREA_DIRECTIONS.get(area_hint, f"the {area_hint.replace('_', ' ')}")
        part_readable  = required_part.replace("_", " ")
        lang_note      = "camera_guidance in Hindi.\n" if language == "hi" else ""

        # ROI hint — narrows Gemini's search to the expected region
        roi_block = ""
        if search_roi is not None:
            roi_cx, roi_cy, roi_margin = search_roi
            roi_block = (
                f"\nROI HINT: The part was last seen near (cx={roi_cx:.2f}, cy={roi_cy:.2f}) "
                f"in normalised image coordinates. Focus your search within "
                f"±{roi_margin:.2f} of that position before searching elsewhere.\n"
            )

        prompt = f"""\
Machine: {machine_type}. Camera is searching for: "{part_readable}", expected in \
{area_hint} ({area_desc}).
{lang_note}{roi_block}
STEP 1 — Look honestly. Name the single most prominent identifiable component in \
frame (shape/colour/position), even if it is NOT the target. Never claim to see the \
target if what you see is actually something else.

STEP 2 — Decide is_target:
  true  → the visible component matches "{part_readable}" in shape AND function AND \
sits in {area_hint} (not merely similar-looking, and not a different port/cap/pipe \
that shares the same rough shape).
  false → anything else: nothing identifiable, wrong section, too blurry/dark, \
likely on another face of the machine, or only part_visibility_pct < 60.

STEP 3 — If is_target=false, use your general mechanical knowledge of a \
{machine_type}'s layout to reason from whatever IS visible toward the target, then \
write ONE imperative sentence (<15 words) in camera_guidance:
  "This is the [visible component]. The {part_readable} is [direction/relation] of it."
  Nothing recognisable       → "Point camera at {area_direction}."
  Likely on other face       → "Turn the {machine_type} around — {part_readable} is on the other side."
  Blurry/dark                → "Hold still and move closer — image unclear."
  Glare on visible surface   → "Tilt phone slightly to reduce reflection."
If is_target=true, camera_guidance = "".

BBOX RULES (fill in ONLY when is_target=true):
  bbox = [cx, cy, w, h], ALL values normalised strictly between 0.0 and 1.0.
  cx/cy = centre (0=left/top, 1=right/bottom). Never return pixel coordinates.
  If a computed value is > 1.0, divide by image dimensions first.
  part_visibility_pct: integer 0–100, how much of the part is unobstructed.
  If part_visibility_pct < 60, set is_target=false and bbox=null instead.

Return ONLY this JSON (no markdown, no preamble):
{{
  "visible_component": "<what you actually see, honestly>",
  "is_target": true|false,
  "bbox": [cx, cy, w, h] or null,
  "confidence": 0.0-1.0,
  "part_description": "<colour, shape, exact location — only if is_target>",
  "camera_guidance": "<STEP 3 sentence, or empty string if is_target=true>",
  "part_visibility_pct": 0-100,
  "part_occluded": true|false,
  "image_too_blurry": true|false,
  "part_behind_machine": true|false
}}"""

        response_text = await vision_call(
            prompt=prompt,
            image_bytes=resized_bytes,
            max_tokens=400,
            temperature=0.1,
        )

        raw = repair_json(response_text)

        # ── Anti-hallucination gate ───────────────────────────────────────────
        # is_target is Gemini's own honest identity verdict (see STEP 2 of the
        # prompt). We never trust "is_target" alone for the bbox — visibility,
        # occlusion and confidence are re-checked independently below — but we
        # DO trust it to decide whether a bbox may exist at all. Crucially we
        # no longer collapse "wrong part" into a single boolean that erases
        # Gemini's reasoning: visible_component + camera_guidance survive
        # regardless of is_target, so the farmer always gets Gemini's actual
        # spatial reasoning rather than a generic fallback string.
        vis_pct = int(raw.get("part_visibility_pct", 100))

        any_rejected = (
            not raw.get("is_target", False)
            or raw.get("part_occluded",       False)
            or raw.get("image_too_blurry",    False)
            or raw.get("part_behind_machine", False)
            or vis_pct < 60
        )
        if vis_pct < 60 and raw.get("is_target", False):
            logger.info(f"locate_part: rejected — visibility={vis_pct}% < 60%")

        raw_found = raw.get("is_target", False) and not any_rejected
        raw_conf  = float(raw.get("confidence") or 0.0)

        # Second gate: confidence below threshold → not found
        if raw_conf < _CONF_THRESHOLD:
            raw_found = False

        bbox = None
        if raw_found:
            raw_bbox = raw.get("bbox")
            if (isinstance(raw_bbox, list) and len(raw_bbox) == 4
                    and all(isinstance(v, (int, float)) for v in raw_bbox)):
                cx, cy, w, h = (float(v) for v in raw_bbox)

                # FIX 5: Strict coordinate validation — all values strictly in (0,1)
                # LLMs sometimes return -0.1, 1.2, or 0.0 (edge) — all rejected.
                coords_ok = (0.0 < cx < 1.0 and 0.0 < cy < 1.0
                             and 0.0 < w < 1.0 and 0.0 < h < 1.0)

                # FIX 9: Size sanity — part must fill 3%–60% per axis,
                # and 2%–60% of total image area.
                area_frac = w * h
                # Area threshold lowered 0.02 → 0.009:
                # clutch_pedal w=0.10 h=0.18 area=0.018 was rejected.
                # Individual w/h >= 0.03 already prevent degenerate boxes.
                # Per-axis upper limit REMOVED — any single-axis width is valid.
                # Reason: air_filter housing fills 90% of frame height (h=0.90),
                # clutch_pedal fills 65–70% of width — both are real detections.
                # A per-axis cap causes false rejections for large parts.
                # Area max raised 0.60 → 0.70: covers w=0.70 h=0.90 (area=0.63).
                # Full-image hallucinations (area > 0.70) are still blocked.
                size_ok   = (0.03 <= w and 0.03 <= h
                             and 0.009 <= area_frac <= 0.70)

                # Bbox must not spill outside image AND not touch the edge
                # (edge-touching bbox usually means part is cut off or machine
                # is partially out of frame → unreliable detection)
                # Strict bounds: bbox must stay within 0.0–1.0 (no margin).
                # A 0.01 edge margin was rejecting parts that touch the frame
                # edge — the UX 'move back' hint handles these cases instead.
                bounds_ok = (
                    cx - w / 2 >= 0.0 and
                    cx + w / 2 <= 1.0 and
                    cy - h / 2 >= 0.0 and
                    cy + h / 2 <= 1.0
                )

                if coords_ok and size_ok and bounds_ok:
                    bbox = [cx, cy, w, h]
                else:
                    raw_found = False
                    bbox = None
                    logger.warning(
                        f"⚠️  locate_part bbox rejected "
                        f"coords={coords_ok} size={size_ok} bounds={bounds_ok} "
                        f"cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}"
                    )

        # Camera guidance — Gemini is asked to always fill this in (STEP 3 of the
        # prompt) whenever is_target=false, using its own spatial reasoning about
        # whatever IS visible. We only fall back to the generic deterministic
        # string if Gemini genuinely returned nothing (empty/missing field) —
        # e.g. a malformed response — never as a routine replacement for its
        # reasoning.
        guidance = raw.get("camera_guidance") or _default_guidance(
            required_part, area_hint, machine_type,
            raw.get("part_behind_machine", False),
            False,  # wrong_area no longer a discrete flag — folded into is_target
            raw.get("image_too_blurry", False),
            language,
        )

        result = {
            "found":               raw_found,
            "bbox":                bbox,
            "confidence":          round(raw_conf, 3) if raw_found else 0.0,
            "part_description":    raw.get("part_description") if raw_found else None,
            "visible_component":   raw.get("visible_component"),  # what Gemini actually saw
            "camera_guidance":     guidance,
            "part_visibility_pct": vis_pct,
            "frame_id":            frame_id,  # echoed back so Flutter can discard stale
            "reject_flags": {
                "part_not_visible":    not raw.get("is_target", False),
                "image_too_blurry":    raw.get("image_too_blurry",    False),
                "part_behind_machine": raw.get("part_behind_machine", False),
                "part_occluded":       raw.get("part_occluded",       False),
            },
        }

        flag_log = [k for k, v in result["reject_flags"].items() if v]
        logger.info(
            f"{'✅' if raw_found else '❌'} locate_part: found={raw_found} "
            f"conf={raw_conf:.2f} flags={flag_log or 'none'} "
            f"visible='{result['visible_component']}' guidance='{guidance}'"
        )
        # Cache successful detections so subsequent calls within 1.5 s skip Gemini
        if raw_found:
            _bbox_cache[_cache_key] = {"result": dict(result), "ts": _now_ts, "hits": 0}
            logger.debug(f"\U0001f4be locate_part cached key={_cache_key}")
        return result

    except json.JSONDecodeError as exc:
        logger.error(f"❌ locate_part JSON error: {exc}")
    except Exception as exc:
        logger.error(f"❌ locate_part error: {exc}")

    return _fallback_not_found(required_part, area_hint, machine_type, language)


def _default_guidance(
    part: str,
    area_hint: str,
    machine: str,
    behind: bool,
    wrong_area: bool,
    blurry: bool,
    lang: str,
) -> str:
    """Deterministic fallback guidance — never empty, never hallucinates."""
    lang = lang or "en"
    part_r = part.replace("_", " ")
    area_r = _AREA_DIRECTIONS.get(area_hint, area_hint.replace("_", " "))

    if blurry:
        return ("कैमरा स्थिर रखें और करीब लाएं" if lang == "hi"
                else "Hold still and move closer — camera is out of focus")
    if behind:
        return (f"मशीन को पलटें — {part_r} पीछे की तरफ है" if lang == "hi"
                else f"Turn the {machine} around — {part_r} is on the other side")
    if wrong_area:
        return (f"कैमरा {area_r} की तरफ करें" if lang == "hi"
                else f"Point camera at {area_r}")
    return (f"{part_r} ढूंढने के लिए कैमरा {area_r} पर लाएं" if lang == "hi"
            else f"Move camera to {area_r} to find the {part_r}")


def _fallback_not_found(part: str, area: str, machine: str, lang: str) -> dict:
    return {
        "found":             False,
        "bbox":              None,
        "confidence":        0.0,
        "part_description":  None,
        "visible_component": None,
        "camera_guidance":   _default_guidance(part, area, machine, False, False, False, lang),
        "reject_flags":      {"part_not_visible": True, "image_too_blurry": False,
                              "part_behind_machine": False, "part_occluded": False},
    }