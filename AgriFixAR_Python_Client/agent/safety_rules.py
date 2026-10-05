"""
agent/safety_rules.py
Deterministic (non-LLM) safety constraints applied BEFORE and AFTER
Gemini reasoning. These rules can NEVER be overridden by prompt injection.

Now machine-aware: critical parts and fuel-system parts are looked up
from the machine_registry for every supported machine type.
"""

from __future__ import annotations
import logging
from typing import Optional

from agent.models import RepairSession, AgentNextResponse
from utils.machine_registry import (
    get_critical_parts,
    get_fuel_system_parts,
    is_electric_machine,
)
from services.safety_guards import (
    _guard_emergency_hazard,
    _guard_electric_hazard,
    _guard_dangerous_workaround,
)

logger = logging.getLogger(__name__)

# Maximum damaged parts before auto-escalate (applies to all machines)
MAX_DAMAGED_BEFORE_ESCALATE = 3


# ─────────────────────────────────────────────
# Pre-check: run BEFORE calling Gemini
# ─────────────────────────────────────────────

def pre_check(session: RepairSession) -> Optional[AgentNextResponse]:
    """
    Return a forced AgentNextResponse if a hard-safety rule fires.
    Returns None if all clear (proceed to Gemini).
    """
    machine_type = session.machine_type
    damaged = [p for p, s in session.verified_parts.items() if s == "damaged"]

    # Fetch machine-specific critical and fuel parts from registry
    critical_parts = set(get_critical_parts(machine_type))
    fuel_parts = set(get_fuel_system_parts(machine_type))
    is_electric = is_electric_machine(machine_type)

    # Rule 1 – critical internal damage detected → escalate immediately
    critical_damaged = set(damaged) & critical_parts
    if critical_damaged:
        logger.warning(f"🚨 Critical parts damaged on {machine_type}: {critical_damaged}")
        return _escalate_response(
            f"Critical component(s) damaged on {machine_type}: {', '.join(critical_damaged)}. "
            "Internal damage confirmed — professional service required.",
            session,
        )

    # Rule 2 – too many damaged parts → escalate
    if len(damaged) >= MAX_DAMAGED_BEFORE_ESCALATE:
        logger.warning(f"🚨 Too many damaged parts ({len(damaged)}) on {machine_type} — escalating")
        return _escalate_response(
            f"{len(damaged)} components found damaged on {machine_type}. "
            "Repair complexity exceeds field service scope — escalate to authorised workshop.",
            session,
        )

    # Rule 3 – electric machine: live-wire step attempted
    # (handled in post_check below; pre_check only logs)
    if is_electric and len(damaged) > 0:
        logger.info(f"⚡ Electric machine {machine_type} — live-wire safety rules active")

    # Rule 4 – fuel leak unresolved (logged, enforced in post_check)
    fuel_damaged = set(damaged) & fuel_parts
    if fuel_damaged:
        logger.info(f"⚠️  Fuel system damage on {machine_type}: {fuel_damaged} — ignition steps will be blocked")

    return None  # all clear


# ─────────────────────────────────────────────
# Post-check: run AFTER Gemini returns
# ─────────────────────────────────────────────

def post_check(response: AgentNextResponse, session: RepairSession) -> AgentNextResponse:
    """
    Validate / sanitise Gemini's proposed next step.
    Mutates response in-place and returns it.
    """
    machine_type = session.machine_type
    damaged = {p for p, s in session.verified_parts.items() if s == "damaged"}
    already_ok = {p for p, s in session.verified_parts.items() if s == "ok"}
    fuel_parts = set(get_fuel_system_parts(machine_type))
    is_electric = is_electric_machine(machine_type)

    # Rule A – Do not re-check an already-verified healthy part
    if response.next_step.required_part in already_ok:
        logger.warning(
            f"⚠️  Gemini tried to re-check {response.next_step.required_part} (already OK) "
            f"on {machine_type}."
        )
        note = (
            f"NOTE: {response.next_step.required_part} was already verified OK. "
            "Skipping duplicate check — consult a mechanic if problem persists."
        )
        response.next_step.safety_warning = note

    # Rule B – Fuel system damaged → block any ignition / cranking step
    ignition_parts = {"ignition_key", "spark_plug", "starter_motor", "glow_plug", "fuel_tap"}
    if damaged & fuel_parts and response.next_step.required_part in ignition_parts:
        logger.warning(
            f"🚨 Post-check: ignition step requested while fuel leak active on {machine_type} — UNSAFE"
        )
        response.status = "unsafe"
        response.next_step.safety_warning = (
            "UNSAFE: Fuel system damage has been detected. "
            "Do NOT start the engine or test ignition until the fuel leak is fully repaired. "
            "Move away from the machine immediately."
        )

    # Rule C – Electric machine: inject power-off reminder on every step
    if is_electric:
        base = response.next_step.safety_warning or ""
        if "power" not in base.lower() and "switch off" not in base.lower():
            response.next_step.safety_warning = (
                "Switch OFF the main power supply and verify with a test lamp before touching any part. "
                + base
            ).strip()

    # Rule D – Engine-off reminder for all non-electric machines on first step
    elif session.current_stage == 0:
        base = response.next_step.safety_warning or ""
        if "engine off" not in base.lower() and "engine is off" not in base.lower():
            response.next_step.safety_warning = (
                "Ensure the engine is completely OFF and the key is removed before proceeding. "
                + base
            ).strip()

    # Rule E – Chaff cutter: safety guard warning always injected
    if machine_type == "chaff_cutter":
        base = response.next_step.safety_warning or ""
        if "guard" not in base.lower() and "feed" not in base.lower():
            response.next_step.safety_warning = (
                "CRITICAL: Verify the safety guard is in place over the feed opening before proceeding. "
                "NEVER put hands inside the feed inlet. " + base
            ).strip()

    return response


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _escalate_response(reason: str, session: RepairSession) -> AgentNextResponse:
    from agent.models import NextStepDetail, UpdatedMemory
    return AgentNextResponse(
        status="escalate",
        reasoning_summary=reason,
        next_step=NextStepDetail(
            text=reason,
            text_en=reason,
            text_hi="यह मरम्मत किसी अधिकृत मैकेनिक को दिखाएं।",
            visual_cue="none",
            ar_model="none.obj",
            required_part="none",
            area_hint="engine_compartment",
            safety_warning="Stop all repair attempts and contact a certified technician immediately.",
        ),
        updated_memory=UpdatedMemory(
            verified_parts=dict(session.verified_parts),
            diagnostic_path=session.diagnostic_path + ["escalated"],
        ),
    )
# ─────────────────────────────────────────────────────────────
# P0-1: Text hazard check — reuses /diagnose guards in agent
# ─────────────────────────────────────────────────────────────

def text_hazard_check(session: RepairSession):
    """
    Run the same deterministic text-pattern hazard guards that protect
    /diagnose against the agent session's free text. The agent normally
    bypasses these entirely — this closes that gap without duplicating
    any regex logic.

    Checks both session.problem (original complaint) and the latest
    ai_observation text (hazards spotted mid-repair via camera).

    Returns an AgentNextResponse if a hazard blocks the step, or None.
    """
    machine_type = session.machine_type
    machine_label = machine_type  # guards only use this for display text

    # Gather all free text in the session
    texts = [session.problem]
    if hasattr(session, 'verified_observations'):
        texts.extend(session.verified_observations.values())

    for text in texts:
        if not text or not isinstance(text, str):
            continue

        # Guard 1: Emergency hazard patterns (fire, arc, entanglement, etc.)
        hit = _guard_emergency_hazard(text)
        if hit is not None:
            return _hazard_dict_to_agent_response(hit, session)

        # Guard 2: Electric shock / flood patterns
        hit = _guard_electric_hazard(text, machine_type)
        if hit is not None:
            return _hazard_dict_to_agent_response(hit, session)

        # Guard 3: Dangerous workaround patterns
        hit = _guard_dangerous_workaround(text)
        if hit is not None:
            return _hazard_dict_to_agent_response(hit, session)

    return None


def _hazard_dict_to_agent_response(hazard: dict, session: RepairSession):
    """Convert a diagnosis-service hazard dict into an AgentNextResponse."""
    from agent.models import NextStepDetail, UpdatedMemory

    warnings_en = hazard.get("safety_warnings_en", [])
    warnings_hi = hazard.get("safety_warnings_hi", [])
    default_en = "Stop. Unsafe condition detected. Do not proceed."
    default_hi = "रुकें। असुरक्षित स्थिति पाई गई। आगे न बढ़ें।"

    return AgentNextResponse(
        status="escalate",
        reasoning_summary=hazard.get(
            "technical_analysis",
            "Hazard detected by safety system."
        ),
        next_step=NextStepDetail(
            text=warnings_en[0] if warnings_en else default_en,
            text_en=warnings_en[0] if warnings_en else default_en,
            text_hi=warnings_hi[0] if warnings_hi else default_hi,
            visual_cue="warning",
            ar_model="none.obj",
            required_part="none",
            area_hint="full_machine",
            safety_warning=warnings_en[0] if warnings_en else default_en,
        ),
        updated_memory=UpdatedMemory(
            verified_parts=dict(session.verified_parts),
            diagnostic_path=session.diagnostic_path + ["hazard_escalated"],
        ),
    )