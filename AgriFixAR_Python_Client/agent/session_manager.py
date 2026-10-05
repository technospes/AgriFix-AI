"""
agent/session_manager.py
Thread-safe in-memory store for active repair sessions.

Design notes:
  - Single process (Uvicorn workers=1 on HuggingFace) → plain dict is safe.
  - For multi-worker deployments replace with Redis / external store.
  - Sessions expire after SESSION_TTL_SECONDS of inactivity.
"""

from __future__ import annotations
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

from agent.models import RepairSession
from agent.validation import validate_repair_plan_steps

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
SESSION_TTL_SECONDS = 3600  # 1 hour inactivity → session dropped

# ── Store ────────────────────────────────────────────────────────────────────
_sessions: Dict[str, RepairSession] = {}
_last_access: Dict[str, datetime] = {}


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def create_session(machine_type: str, problem: str, language: str = "en",
                   diagnosis_steps: list = None) -> RepairSession:
    session_id = str(uuid.uuid4())
    session = RepairSession(
        session_id=session_id,
        machine_type=machine_type,
        problem=problem,
        language=language,
    )
    if diagnosis_steps:
        from agent.models import RepairPlan, RepairPlanStep

        # step_id is owned exclusively by diagnosis_service.py (the single
        # source of truth — see agent/validation.py). This function must only
        # COPY ids that already arrived on diagnosis_steps, never invent one
        # via a fallback default: a silent f"s{i+1}" here would mask a caller
        # that skipped diagnosis_service.py (or a tampered/malformed client
        # payload) exactly the way the original step_id bug did. Validate
        # up front and raise InvalidRepairPlan — a backend defect, not a
        # mechanical fault — instead of quietly patching it.
        validate_repair_plan_steps(diagnosis_steps, context=f"machine={machine_type} (session creation)")

        steps = [
            RepairPlanStep(
                step_id=s.get("step_id"),
                action=s.get("action", ""),
                description=s.get("description", ""),
                required_part=s.get("required_part", "unknown"),
                area_hint=s.get("area_hint", "engine_compartment"),
                step_type=s.get("step_type", "inspection"),
            )
            for s in diagnosis_steps
        ]
        session.repair_plan = RepairPlan(
            machine_type=machine_type,
            steps=steps,
        )
        session.current_step_id = steps[0].step_id
    _sessions[session_id] = session
    _last_access[session_id] = datetime.utcnow()
    logger.info(f"🆕 Session created: {session_id}  machine={machine_type}  steps={len(steps) if diagnosis_steps else 0}")
    return session

def get_session(session_id: str) -> Optional[RepairSession]:
    """Return session or None if not found / expired."""
    _evict_expired()
    session = _sessions.get(session_id)
    if session:
        _last_access[session_id] = datetime.utcnow()
    return session


def update_session(session: RepairSession) -> None:
    """Persist updated session back to the store."""
    _sessions[session.session_id] = session
    _last_access[session.session_id] = datetime.utcnow()
    logger.debug(f"💾 Session updated: {session.session_id}  stage={session.current_stage}")


def delete_session(session_id: str) -> bool:
    """Explicitly remove a session (e.g. after resolution)."""
    existed = session_id in _sessions
    _sessions.pop(session_id, None)
    _last_access.pop(session_id, None)
    if existed:
        logger.info(f"🗑️  Session deleted: {session_id}")
    return existed


def list_sessions() -> list[str]:
    """Return all active session IDs (debug / admin use)."""
    _evict_expired()
    return list(_sessions.keys())


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _evict_expired() -> None:
    """Remove sessions that have been idle beyond TTL."""
    cutoff = datetime.utcnow() - timedelta(seconds=SESSION_TTL_SECONDS)
    expired = [sid for sid, ts in _last_access.items() if ts < cutoff]
    for sid in expired:
        _sessions.pop(sid, None)
        _last_access.pop(sid, None)
        logger.info(f"⏰ Session expired + evicted: {sid}")