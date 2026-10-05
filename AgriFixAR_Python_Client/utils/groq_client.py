"""
utils/groq_client.py
Shared Groq client — single source of truth for all LLM text-generation calls.

All text-generation modules (diagnosis_service, multihop_diagnosis, query_router,
repair_agent, etc.) must import from here. Never instantiate Groq() elsewhere.

VISION MODEL NOTE:
  Visual inference (visual_gate.py, locate_part_service.py, verification_service.py)
  uses llama-3.2-11b-vision-preview and will be migrated in a SEPARATE future task.
  Do NOT use this client for vision calls yet.
"""  # MIGRATED: Gemini → Groq

from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()
from groq import Groq  # MIGRATED: Gemini → Groq

logger = logging.getLogger(__name__)

# ── API key rotation: round-robin with rate-limit cooldown ───────────────────
# Loads up to 4 keys from env. Distributes traffic evenly via round-robin.
# Rate-limited keys are temporarily skipped to avoid wasting attempts.

import threading
import time as _time_module

_GROQ_API_KEYS = []
for _env_var in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4"):
    _key = os.environ.get(_env_var)
    if _key:
        _GROQ_API_KEYS.append(_key)

if not _GROQ_API_KEYS:
    raise ValueError("GROQ_API_KEY missing — set at least one in .env")

_groq_clients = [Groq(api_key=k) for k in _GROQ_API_KEYS]
groq_client = _groq_clients[0]  # default client for backward compatibility
_CLIENT_TO_INDEX = {id(c): i for i, c in enumerate(_groq_clients)}

# Round-robin state: current index + per-key cooldown timestamps
_KEY_INDEX = -1
_KEY_COOLDOWN: dict[int, float] = {}  # key_index → unix timestamp when usable again
_KEY_LOCK = threading.Lock()
_RATE_LIMIT_COOLDOWN_SECONDS = float(os.getenv("GROQ_KEY_COOLDOWN_SECONDS", "30"))


def _next_client() -> Groq:
    """
    Return the next available Groq client using round-robin with cooldown.

    Skips keys that are in cooldown (recently rate-limited). If all keys
    are in cooldown, returns the one with the earliest expiry.
    Thread-safe: uses a lock to serialize access to the round-robin state.
    """
    global _KEY_INDEX
    with _KEY_LOCK:
        _now = _time_module.monotonic()
        _n = len(_groq_clients)

        # Try each key in round-robin order, skipping cooled-down ones
        for _ in range(_n):
            _KEY_INDEX = (_KEY_INDEX + 1) % _n
            _cooldown_until = _KEY_COOLDOWN.get(_KEY_INDEX, 0)
            if _now >= _cooldown_until:
                return _groq_clients[_KEY_INDEX]

        # All keys in cooldown — pick the one with earliest expiry
        _best_idx = min(_KEY_COOLDOWN, key=_KEY_COOLDOWN.get)
        _KEY_INDEX = _best_idx
        _remaining = _KEY_COOLDOWN[_best_idx] - _now
        if _remaining > 0:
            logger.warning(
                "All %d Groq keys in cooldown — using key %d (%.1fs remaining)",
                _n, _best_idx, _remaining,
            )
        return _groq_clients[_best_idx]


def _mark_key_rate_limited(client: Groq) -> None:
    """Put the key associated with this client into cooldown."""
    _i = _CLIENT_TO_INDEX.get(id(client))
    if _i is not None:
        with _KEY_LOCK:
            _KEY_COOLDOWN[_i] = _time_module.monotonic() + _RATE_LIMIT_COOLDOWN_SECONDS
        logger.info("⏳ Groq key %d rate-limited — cooldown for %.0fs", _i, _RATE_LIMIT_COOLDOWN_SECONDS)


logger.info(
    "Groq client initialised | %d API keys loaded | cooldown=%.0fs",
    len(_GROQ_API_KEYS), _RATE_LIMIT_COOLDOWN_SECONDS,
)

# ── Model identifiers ──────────────────────────────────────────────────────────
TEXT_MODEL = "llama-3.3-70b-versatile"  # MIGRATED: Gemini → Groq
TEXT_MODEL_FALLBACK = "llama-3.1-8b-instant"  # FAILOVER: primary → fallback

# FUTURE: vision model reserved for a separate migration task
# VISION_MODEL = "llama-3.2-11b-vision-preview"  # MIGRATED: Gemini → Groq (future)

# ── Model registry (preparation for future multi-level failover) ──────────────
AVAILABLE_MODELS = [  # FAILOVER: primary → fallback
    TEXT_MODEL,
    TEXT_MODEL_FALLBACK,
]

# ── Default generation config for JSON-producing calls ───────────────────────
# temperature=0.1  → near-deterministic for grounded repair reasoning
# max_tokens=2048  → sufficient for diagnosis JSON + reasoning CoT
JSON_CONFIG = {  # MIGRATED: Gemini → Groq
    "temperature": 0.1,
    "max_tokens": 2048,
}

# ── Lighter config for short-output calls (router, multihop) ──────────────────
SHORT_CONFIG = {  # MIGRATED: Gemini → Groq
    "temperature": 0.1,
    "max_tokens": 400,
}

# ── Recoverable error signals (trigger failover) ──────────────────────────────
_RECOVERABLE_ERRORS = (  # FAILOVER: primary → fallback
    "429",
    "rate limit",
    "rate_limit_exceeded",
    "quota",
    "tokens per day",
    "tpd",
    "tpm",
    "rpm",
    "too many requests",
    "capacity",
    "overloaded",
    "service unavailable",
)

# ── Non-recoverable error signals (raise immediately, no failover) ─────────────
_NON_RECOVERABLE_ERRORS = (  # FAILOVER: primary → fallback
    "authentication",
    "invalid api key",
    "permission denied",
    "invalid request",
    "malformed",
    "bad request",
)


# ── Retry configuration (env-configurable) ────────────────────────────────────
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "2"))
GROQ_BASE_DELAY  = float(os.getenv("GROQ_BASE_DELAY", "0.8"))
GROQ_MAX_DELAY   = float(os.getenv("GROQ_MAX_DELAY", "2.5"))


# ── Error classification helpers ──────────────────────────────────────────────

def _is_fatal(err_str: str) -> bool:
    """True if this error should NOT trigger failover or retry."""
    return any(kw in err_str for kw in _NON_RECOVERABLE_ERRORS)


def _is_retryable(err_str: str) -> bool:
    """True if this error should trigger failover and eventual retry."""
    return any(kw in err_str for kw in _RECOVERABLE_ERRORS)


def _compute_backoff(attempt: int) -> float:
    """Jittered exponential backoff. Returns seconds to sleep."""
    import random
    delay = min(GROQ_BASE_DELAY * (2 ** attempt), GROQ_MAX_DELAY)
    delay += random.uniform(0.2, 0.8)
    return delay


def _single_attempt(
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> Any:
    primary_error: Optional[Exception] = None

    # Round-robin key selection — guarantees fair distribution
    _client = _next_client()
    # ── Attempt primary model ─────────────────────────────────────────────────
    try:
        response = _client.chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.debug("Groq primary model succeeded | model=%s", TEXT_MODEL)
        return response

    except Exception as e:
        err_str = str(e).lower()

        if _is_fatal(err_str):
            logger.error("Groq fatal error | model=%s error=%s", TEXT_MODEL, e)
            raise

        # Only cooldown on actual quota/rate-limit errors, not transient failures
        if any(kw in err_str for kw in ("429", "rate limit", "tpd", "tpm", "rpm", "quota", "tokens per day", "too many requests")):
            _mark_key_rate_limited(_client)

        logger.warning(
            "Groq primary failed; attempting fallback | model=%s error=%s",
            TEXT_MODEL, e,
        )
        primary_error = e

    # ── Token-saving fallback ────────────────────────────────────────────────
    fallback_max_tokens = max_tokens
    if primary_error is not None:
        primary_err_str = str(primary_error).lower()
        if "tokens per day" in primary_err_str or "tpd" in primary_err_str:
            fallback_max_tokens = min(max_tokens, 1200)
            logger.info(
                "FAILOVER: TPD limit hit — capping fallback max_tokens to %d",
                fallback_max_tokens,
            )

    _fb_client = _next_client()
    try:
        response = _fb_client.chat.completions.create(
            model=TEXT_MODEL_FALLBACK,
            messages=messages,
            temperature=temperature,
            max_tokens=fallback_max_tokens,
        )
        logger.info("Groq fallback model used successfully | model=%s", TEXT_MODEL_FALLBACK)
        return response

    except Exception as fallback_error:
        fb_err_str = str(fallback_error).lower()
        # Cooldown the fallback key if it was also rate-limited
        if any(kw in fb_err_str for kw in ("429", "rate limit", "tpd", "tpm", "rpm", "quota", "tokens per day", "too many requests")):
            _mark_key_rate_limited(_fb_client)
        logger.warning(
            "Groq fallback failed | primary_error=%s fallback_error=%s",
            primary_error, fallback_error,
        )
        raise

# ── Public API: single-attempt failover with configurable retry ───────────────

def groq_chat_completion(
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Any:
    """
    Centralized Groq completion with automatic failover + retry + jitter.

    This function is synchronous. Callers from async code MUST invoke it
    via asyncio.to_thread().

    Flow per attempt:
        1. Try TEXT_MODEL → fallback to TEXT_MODEL_FALLBACK.
        2. If both fail with a retryable error, sleep with jittered backoff
           and retry up to GROQ_MAX_RETRIES times.
        3. On non-recoverable failure, raise immediately (no retry).

    Retry parameters are configured via environment variables:
        GROQ_MAX_RETRIES  (default 2)
        GROQ_BASE_DELAY   (default 0.8 seconds)
        GROQ_MAX_DELAY    (default 2.5 seconds)

    Uses the module-level groq_client — never creates a new client per call.

    Returns:
        The raw Groq completion response object.
    """
    import time

    last_error: Optional[Exception] = None

    for attempt in range(GROQ_MAX_RETRIES + 1):
        try:
            response = _single_attempt(messages, temperature, max_tokens)
            if attempt > 0:
                logger.info(
                    "Groq succeeded on retry %d/%d",
                    attempt, GROQ_MAX_RETRIES,
                )
            return response

        except Exception as e:
            err_str = str(e).lower()
            last_error = e

            # Fatal errors — no retry, raise immediately
            if _is_fatal(err_str):
                logger.error(
                    "Groq fatal error (attempt %d/%d) | error=%s",
                    attempt + 1, GROQ_MAX_RETRIES + 1, e,
                )
                raise

            # Retryable — sleep with jittered backoff, then retry
            if attempt < GROQ_MAX_RETRIES:
                delay = _compute_backoff(attempt)
                logger.warning(
                    "Groq attempt %d/%d failed; retrying in %.1fs | error=%s",
                    attempt + 1, GROQ_MAX_RETRIES + 1, delay, e,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Groq exhausted all attempts (%d) | last error=%s",
                    GROQ_MAX_RETRIES + 1, last_error,
                )

    raise last_error  # type: ignore[misc]


logger.info(
    "Groq client initialised | text_model=%s fallback_model=%s "
    "retries=%d base_delay=%.1fs max_delay=%.1fs",
    TEXT_MODEL, TEXT_MODEL_FALLBACK,
    GROQ_MAX_RETRIES, GROQ_BASE_DELAY, GROQ_MAX_DELAY,
)