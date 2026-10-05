"""
db_lock.py — AgriFix Database-Only Lock v1.0
=============================================
Phase 2: Guarantee the system NEVER invents repair steps.
"""

from __future__ import annotations

import logging
import os
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger(__name__)

LOCK_SCORE_THRESHOLD: float = float(os.environ.get("AGRIFIX_LOCK_THRESHOLD", "0.35"))
MIN_CHUNKS_REQUIRED: int = int(os.environ.get("AGRIFIX_MIN_CHUNKS", "1"))

# ── Safe Windows SQLite execution ──────────────────────────────────────────────

def retry_with_backoff(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Windows-safe retry wrapper for SQLite / ChromaDB operations.
    Prevents 'database is locked' deadlocks using randomized exponential backoff.
    """
    max_retries = 5
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except PermissionError as e:  # Common Windows file lock error
            if i == max_retries - 1:
                raise
            sleep_time = (0.1 * (2 ** i)) + (random.uniform(0, 0.1))
            logger.warning(f"DB busy (PermissionError), retrying in {sleep_time:.2f}s... ({e})")
            time.sleep(sleep_time)
        except Exception as e:
            if "database is locked" in str(e).lower() or "busy" in str(e).lower():
                if i == max_retries - 1:
                    raise
                sleep_time = (0.1 * (2 ** i)) + (random.uniform(0, 0.1))
                logger.warning(f"Database busy/locked, retrying in {sleep_time:.2f}s... ({e})")
                time.sleep(sleep_time)
            else:
                raise

# ── Lock result model ─────────────────────────────────────────────────────────

@dataclass
class DbLockResult:
    locked: bool
    score: float
    n_chunks: int
    reason: str
    machine_type: str
    query: str

    def api_response(self) -> Dict[str, Any]:
        return {
            "status": "no_data",
            "locked": True,
            "machine_type": self.machine_type,
            "diagnosis": (
                "I do not have the certified repair steps for this specific issue "
                "in my database. Please contact a qualified technician or refer to "
                "your machine's official service manual."
            ),
            "diagnosis_hi": (
                "मेरे डेटाबेस में इस समस्या के लिए प्रमाणित मरम्मत के चरण उपलब्ध नहीं हैं। "
                "कृपया किसी प्रमाणित मैकेनिक से संपर्क करें या अपनी मशीन की आधिकारिक सर्विस मैनुअल देखें।"
            ),
            "confidence_score": round(self.score, 3),
            "lock_reason": self.reason,
            "steps": [],
            "parts": [],
            "safety_warnings": [],
        }

    def log(self) -> None:
        if self.locked:
            logger.warning("🔒 DB LOCK: machine=%s score=%.3f chunks=%d reason='%s' query='%s...'",
                self.machine_type, self.score, self.n_chunks, self.reason, self.query[:60])
        else:
            logger.info("✅ DB PASS: machine=%s score=%.3f chunks=%d",
                self.machine_type, self.score, self.n_chunks)

def check_db_lock(
    score: float,
    n_chunks: int,
    machine_type: str,
    query: str,
    lock_threshold: Optional[float] = None,
    min_chunks: Optional[int] = None,
) -> DbLockResult:
    threshold = lock_threshold if lock_threshold is not None else LOCK_SCORE_THRESHOLD
    min_c     = min_chunks if min_chunks is not None else MIN_CHUNKS_REQUIRED

    if n_chunks < min_c:
        result = DbLockResult(True, score, n_chunks, f"No database chunks found for machine='{machine_type}'", machine_type, query)
        result.log()
        return result

    if score < threshold:
        result = DbLockResult(True, score, n_chunks, f"Best match score {score:.3f} below lock threshold {threshold:.3f}", machine_type, query)
        result.log()
        return result

    result = DbLockResult(False, score, n_chunks, "", machine_type, query)
    result.log()
    return result

def check_db_lock_from_rag(
    rag_result: tuple,
    machine_type: str,
    query: str,
    lock_threshold: Optional[float] = None,
) -> DbLockResult:
    context_str, score, n_chunks = rag_result
    return check_db_lock(score, n_chunks, machine_type, query, lock_threshold)