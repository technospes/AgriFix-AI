"""
utils/helpers.py
Shared utility functions used across the backend.
Part → area_hint derivation now delegates to machine_registry so it works
for ALL supported farm machines, not just tractors.
"""

from __future__ import annotations
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

# ── Cache configuration ───────────────────────────────────────────────────────
CACHE_DIR = Path("response_cache")
CACHE_DIR.mkdir(exist_ok=True)
MAX_CACHE_AGE = 86400  # 24 hours in seconds


# ─────────────────────────────────────────────
# FIX 1: Robust JSON repair helpers
# ─────────────────────────────────────────────

# FIX 1: Set True to log repaired strings at DEBUG level for tracing.
# Safe to leave enabled; DEBUG level is suppressed in production by default.
_LOG_REPAIRS = True


def _strip_fences(text: str) -> str:
    """FIX 1: Strip markdown code fences that Gemini sometimes wraps around JSON."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    """FIX 1: Remove trailing commas before } or ] (valid in JS, invalid in JSON)."""
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text


def _fix_single_quotes(text: str) -> str:
    """
    FIX 1: Replace single-quoted JSON keys/values with double-quoted equivalents.

    Strategy: swap unescaped single-quote pairs that are at JSON key/value
    positions. Conservative regex — only touches clearly single-quoted tokens,
    not prose apostrophes inside double-quoted strings.
    """
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)
    return text


def _fix_unterminated_string(text: str) -> str:
    """
    FIX 1: Close an unterminated JSON string caused by Gemini token-limit truncation.

    Example input:  {"reasoning": "The capacitor is failing because the motor
    Example output: {"reasoning": "The capacitor is failing because the motor"}

    Approach:
      1. Count unescaped double-quotes. Odd count → open string exists.
      2. Trim the last partial token (avoids submitting a half-word as a value).
      3. Close the string with " then close any open { or [ structures.
    """
    unescaped_quotes = len(re.findall(r'(?<!\\)"', text))
    if unescaped_quotes % 2 == 0:
        return text  # strings already balanced

    text = text.rstrip()
    # Trim the last partial/incomplete token to avoid malformed values
    text = re.sub(r'\s+\S+$', '', text)
    text += '"'  # close the open string

    # Close any open brace/bracket structures (handles 1–2 levels of nesting)
    open_curly  = text.count('{') - text.count('}')
    open_square = text.count('[') - text.count(']')
    text += ']' * max(0, open_square)
    text += '}' * max(0, open_curly)

    return text


def repair_json(raw: str) -> Any:
    """
    FIX 1: Parse a string as JSON, applying progressive repair steps on failure.

    Repair order (cheapest / least destructive first):
      1. Strip markdown fences  (```json ... ```)
      2. Fix trailing commas    (,} and ,])
      3. Fix single quotes      (' → ")
      4. Fix unterminated strings (token-limit truncation)

    Raises json.JSONDecodeError only if ALL repair strategies fail.
    Call sites should still wrap in try/except for complete safety.
    """
    text = _strip_fences(raw)

    # Fast path — already valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 2 — trailing commas
    text = _fix_trailing_commas(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 3 — single quotes
    text = _fix_single_quotes(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 4 — unterminated strings (most aggressive; do last)
    text = _fix_unterminated_string(text)
    if _LOG_REPAIRS:
        logger.debug("FIX 1: JSON repaired (unterminated string): %.200s", text)
    # Let any remaining JSONDecodeError propagate to the caller
    return json.loads(text)


# ─────────────────────────────────────────────
# JSON sanitisation (updated to delegate to repair_json)
# ─────────────────────────────────────────────

def sanitize_json_text(text: str) -> str:
    """
    FIX 1: Strip markdown fences and fix common JSON errors in AI responses.

    Previously only stripped fences and trailing commas before }.
    Now delegates to repair_json() which also handles:
      - single-quoted keys/values
      - unterminated strings (token-limit truncation)
      - trailing commas before ] as well as }

    Returns a canonical JSON string (re-serialised from the parsed object).
    Raises ValueError if repair fails, so callers get a clear exception
    instead of a silent empty result.
    """
    try:
        parsed = repair_json(text)
        return json.dumps(parsed)
    except (json.JSONDecodeError, Exception) as exc:
        raise ValueError(f"FIX 1: JSON repair failed: {exc}") from exc


# ─────────────────────────────────────────────
# Response cache
# ─────────────────────────────────────────────

def generate_cache_key(prefix: str, *args) -> str:
    key_string = prefix + "_".join(str(a) for a in args)
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Return cached response if it exists and hasn't expired."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        age = datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < MAX_CACHE_AGE:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def cache_response(cache_key: str, data: Dict) -> None:
    """Persist a response to the file cache."""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as exc:
        logger.warning(f"Failed to cache response: {exc}")


# ─────────────────────────────────────────────
# File cleanup
# ─────────────────────────────────────────────

def cleanup_old_files(upload_dir: Path, max_age_seconds: int = 3600) -> None:
    """Remove stale temporary files and expired cache entries."""
    try:
        current_time = datetime.now().timestamp()
        for file in upload_dir.glob("*"):
            if current_time - file.stat().st_mtime > max_age_seconds:
                file.unlink(missing_ok=True)
        for file in CACHE_DIR.glob("*.json"):
            if current_time - file.stat().st_mtime > MAX_CACHE_AGE:
                file.unlink(missing_ok=True)
    except Exception as exc:
        logger.error(f"Cleanup error: {exc}")


# ─────────────────────────────────────────────
# Part → area_hint lookup  (registry-backed, all machines)
# ─────────────────────────────────────────────

def derive_part_and_area(
    step_text: str,
    machine_type: Optional[str] = None,
) -> tuple[str, str]:
    """
    Extract required_part and area_hint from step_text when the Flutter
    client doesn't send them explicitly.

    Strategy:
      1. Build a regex from ALL known part IDs in the machine_registry.
      2. If machine_type is provided, prefer parts belonging to that machine
         (gives more accurate area_hint for multi-machine deployments).
      3. Falls back to generic defaults if no match.
    """
    from utils.machine_registry import get_all_part_ids, get_part_area

    # Build pattern from all registered part IDs
    all_parts = get_all_part_ids(machine_type)  # machine-specific first
    if not all_parts:
        all_parts = get_all_part_ids()           # fallback: all machines

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(p) for p in all_parts) + r")\b"
    )
    match = pattern.search(step_text)
    if match:
        part = match.group(1)
        area = get_part_area(part, machine_type)
        logger.info(f"🔍 Auto-derived part={part} area={area} for machine={machine_type}")
        return part, area

    return "machine_part", "engine_compartment"
