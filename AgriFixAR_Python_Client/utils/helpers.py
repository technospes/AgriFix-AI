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
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# ── Cache configuration ───────────────────────────────────────────────────────
CACHE_DIR = Path("response_cache")
CACHE_DIR.mkdir(exist_ok=True)
MAX_CACHE_AGE = 86400  # 24 hours in seconds

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
