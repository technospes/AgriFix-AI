from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import Optional
# MIGRATED: Gemini → Groq — google.generativeai removed
from utils.groq_client import groq_client, TEXT_MODEL, JSON_CONFIG, groq_chat_completion
from utils.json_repair import repair_json
from prompts.renderers.repair import render_repair_prompt, REPAIR_PROMPT_VERSION
from prompts.builder import REPAIR_SYSTEM_BLOCK
from prompts.context import PromptContext
from agent.models import Interaction, InteractionOption, RepairPlanStep, VerificationMode, Verification
from agent.models import RepairSession, AgentNextResponse, NextStepDetail, UpdatedMemory
from agent import safety_rules
from agent.validation import InvalidRepairPlan, validate_repair_plan_steps
from agent.jargon_guard import apply_jargon_guard
from utils.machine_registry import (
    get_profile_or_default,
    get_allowed_area_ids,
    get_compact_parts_list,
    get_compact_diagnostic_hint,
    get_compact_safety_keywords,
    get_critical_parts,
    get_fuel_system_parts,
    is_electric_machine,
)

logger = logging.getLogger(__name__)
# _GEMINI_MODEL removed — TEXT_MODEL from groq_client used instead  # MIGRATED: Gemini → Groq

# ── Machine-specific tool registry ────────────────────────────────────────────
# Prevents hallucinated tools. Only tools confirmed available on-farm per
# machine category are allowed. LLM must pick from this list or set null.
_MACHINE_TOOLS: dict[str, list[str]] = {
    "tractor":          ["multimeter", "spanner_set", "screwdriver_flat",
                         "screwdriver_phillips", "pliers", "wrench_adjustable",
                         "funnel", "clean_cloth", "torch_light"],
    "harvester":        ["spanner_set", "wrench_adjustable", "screwdriver_flat",
                         "pliers", "clean_cloth", "torch_light", "grease_gun"],
    "thresher":         ["spanner_set", "screwdriver_flat", "pliers",
                         "wrench_adjustable", "clean_cloth", "torch_light"],
    "submersible_pump": ["multimeter", "insulated_screwdriver",
                         "rubber_gloves", "torch_light", "pliers"],
    "water_pump":       ["pliers", "spanner_set", "screwdriver_flat",
                         "clean_cloth", "torch_light", "funnel"],
    "electric_motor":   ["multimeter", "insulated_screwdriver",
                         "rubber_gloves", "torch_light", "pliers"],
    "power_tiller":     ["spanner_set", "screwdriver_flat", "pliers",
                         "wrench_adjustable", "clean_cloth", "torch_light"],
    "chaff_cutter":     ["spanner_set", "screwdriver_flat", "pliers",
                         "clean_cloth", "torch_light"],
    "diesel_engine":    ["multimeter", "spanner_set", "screwdriver_flat",
                         "pliers", "wrench_adjustable", "funnel",
                         "clean_cloth", "torch_light"],
    "rotavator":        ["spanner_set", "screwdriver_flat", "pliers",
                         "wrench_adjustable", "clean_cloth", "torch_light"],
    "generator":        ["multimeter", "insulated_screwdriver",
                         "rubber_gloves", "pliers", "torch_light"],
}
_DEFAULT_TOOLS = ["spanner_set", "screwdriver_flat", "pliers", "clean_cloth", "torch_light"]


def _allowed_tools(machine_type: str) -> list[str]:
    return _MACHINE_TOOLS.get(machine_type, _DEFAULT_TOOLS)


def _tools_prompt_block(machine_type: str) -> str:
    tools = _allowed_tools(machine_type)
    return f"ALLOWED TOOLS (choose required_tool from this list or null): {', '.join(tools)}"

def _format_observations(session: RepairSession) -> str:
    if not session.verified_observations:
        return "None yet."
    return "\n".join(
        f"  {part}: {obs}"
        for part, obs in session.verified_observations.items()
    )


def _format_generated_steps(session: RepairSession) -> str:
    if not session.generated_steps:
        return "Not linked to a diagnosis plan."
    done = set(session.verified_parts.keys())
    lines = []
    for entry in session.generated_steps:
        parts = entry.split(":")
        step_id   = parts[0] if len(parts) > 0 else "?"
        part_id   = parts[1] if len(parts) > 1 else "?"
        step_type = parts[2] if len(parts) > 2 else "?"
        status = "✓ done" if part_id in done else "→ pending"
        lines.append(f"  {step_id}: {part_id} ({step_type}) [{status}]")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def _get_area_context(machine_type: str, area_hint: str, language: str) -> dict:
    """Return structured location context from machine registry."""
    from utils.machine_registry import get_area_farmer_description
    
    description = get_area_farmer_description(machine_type, area_hint, language)
    
    # Default landmark map per common area — registry can override later
    _LANDMARKS: dict[str, list[str]] = {
        "motor_housing": ["power cable entry", "cooling fan cover", "capacitor box"],
        "control_panel": ["main switch", "MCB breaker", "terminal board cover"],
        "pump_body": ["suction pipe", "discharge pipe", "priming plug"],
        "engine_compartment": ["fuel filter", "air filter housing", "oil dipstick"],
    }
    
    return {
        "name": area_hint.replace("_", " "),
        "description": description or f"The {area_hint.replace('_', ' ')} area of the machine.",
        "landmarks": _LANDMARKS.get(area_hint, []),
    }

async def decide_next_step(session: RepairSession, last_verification: dict) -> AgentNextResponse:
    _apply_verification(session, last_verification)

    # ── Fail-fast structural validation ─────────────────────────────────────
    # step_id is owned exclusively by diagnosis_service.py, where the repair
    # plan is first built and validated (single source of truth — see
    # agent/validation.py). This agent must never invent or repair a broken
    # plan itself: silently patching it here is exactly what caused the
    # original bug (agent stuck on step 1 forever with no visible error).
    # If an invalid plan somehow still reaches this function, that's a
    # backend defect, not a mechanical fault with the farmer's machine —
    # raise InvalidRepairPlan and let it propagate. The API layer
    # (main.py's /agent/next handler) must catch this separately from
    # normal agent responses and return a generic service-error status,
    # never an "escalate to mechanic" response.
    if session.repair_plan and session.repair_plan.steps:
        try:
            validate_repair_plan_steps(
                session.repair_plan.steps,
                context=f"machine={session.machine_type} session={session.session_id}",
            )
        except InvalidRepairPlan as exc:
            logger.error(
                "❌ [%s] Structurally invalid repair plan reached the agent "
                "(session=%s): %s. Fix this in diagnosis_service.py's "
                "validation, not here.",
                session.machine_type, session.session_id, exc,
            )
            raise

    # O(1) lookup — built once per turn instead of re-scanning the plan
    # (which was O(n) on every branch below, twice, every single call).
    step_map: dict[str, object] = (
        {s.step_id: s for s in session.repair_plan.steps}
        if session.repair_plan else {}
    )
    step_order: list[str] = (
        [s.step_id for s in session.repair_plan.steps]
        if session.repair_plan else []
    )

    # 1. Log BEFORE advancement check
    logger.info("DEBUG: Current step lookup: %r", session.current_step_id)

    next_state = last_verification.get("selected_next_state")
    is_bool_done = last_verification.get("answer_bool") is True

    if next_state == "continue" or is_bool_done:
        current_idx = (
            step_order.index(session.current_step_id)
            if session.current_step_id in step_order else -1
        )
        logger.info("DEBUG: Found current_idx=%d", current_idx)

        if current_idx >= 0 and current_idx + 1 < len(step_order):
            session.current_step_id = step_order[current_idx + 1]
            logger.info("⏩ Advanced to step ID: %r", session.current_step_id)
        elif current_idx >= 0:
            return AgentNextResponse(
                status="resolved",
                reasoning_summary="All steps completed.",
                next_step=NextStepDetail(
                    text="Repair complete.", text_en="All steps done.",
                    text_hi="सभी चरण पूर्ण।", visual_cue="none",
                    ar_model="none.obj", required_part="none",
                    area_hint="engine_compartment",
                ),
                updated_memory=UpdatedMemory(
                    verified_parts=dict(session.verified_parts),
                    diagnostic_path=session.diagnostic_path + ["complete"],
                ),
            )

    # P0-1: Check free text for hazard patterns before any LLM call
    hazard = safety_rules.text_hazard_check(session)
    if hazard is not None:
        logger.info(
            f"🛡️  Text hazard guard blocked agent step "
            f"[{session.machine_type}, session={session.session_id}]"
        )
        return hazard

    forced = safety_rules.pre_check(session)
    if forced:
        logger.info(f"🛡️  Safety pre-check forced response [{session.machine_type}]")
        return forced

    # ── Deterministic step selection from repair plan ──────────────────────
    if not session.repair_plan or not session.repair_plan.steps:
        return _fallback_response(session.machine_type, "No diagnosis plan available")

    # Find current step by ID — O(1) via step_map built above.
    current_step = step_map.get(session.current_step_id)

    if current_step is None:
        return AgentNextResponse(
            status="resolved",
            reasoning_summary="All diagnosis steps completed.",
            next_step=NextStepDetail(
                text="Repair sequence complete.",
                text_en="All diagnosis steps completed. Consult a mechanic if the problem persists.",
                text_hi="सभी जांच चरण पूरे हो गए। समस्या बनी रहे तो मैकेनिक से संपर्क करें।",
                visual_cue="none", ar_model="none.obj", required_part="none",
                area_hint="engine_compartment",
            ),
            updated_memory=UpdatedMemory(
                verified_parts=dict(session.verified_parts),
                diagnostic_path=session.diagnostic_path + ["complete"],
            ),
        )

    profile        = get_profile_or_default(session.machine_type)
    allowed_areas  = " | ".join(get_allowed_area_ids(session.machine_type))
    known_parts    = get_compact_parts_list(session.machine_type)
    safety_context = _build_safety_context(session)
    tools_block    = _tools_prompt_block(session.machine_type)

    area_ctx = _get_area_context(session.machine_type, current_step.area_hint, session.language)

    relevant_parts_list = [current_step.required_part] if current_step.required_part not in (None, "unknown", "") else []
    for part, status in session.verified_parts.items():
        if part not in relevant_parts_list and status in ("damaged", "unclear"):
            relevant_parts_list.append(part)

    verification_cap = {
        "camera_available": True,
        "vision_models": ["locate_part", "verify_step", "inspect_part"],
        "allowed_interactions": ["camera", "boolean", "choice", "number", "none"],
        "step_mode_hint": current_step.verification.mode if current_step.verification else "confirmation"
    }

    ctx = PromptContext(
        machine_type=session.machine_type,
        action=current_step.action,
        description=current_step.description,
        required_part=current_step.required_part or "",
        area_hint=current_step.area_hint,
        area_description=area_ctx["description"],
        area_landmarks=", ".join(area_ctx["landmarks"]) if area_ctx["landmarks"] else "none specified",
        step_type=current_step.step_type,
        attempt_count=session.attempt_count,
        verified_parts_json=json.dumps(session.verified_parts, indent=2),
        visual_observations=_format_observations(session),
        last_verification_json=json.dumps(last_verification, indent=2),
        safety_context=safety_context,
        relevant_areas=current_step.area_hint,
        relevant_parts=", ".join(relevant_parts_list) if relevant_parts_list else "none",
        tools_block=tools_block,
        verification_capability=json.dumps(verification_cap)
    )

    prompt = render_repair_prompt(ctx)
    
    raw = await _call_gemini(prompt)
    response = _parse_response(raw, session.machine_type)
    response = safety_rules.post_check(response, session)

    # ── Jargon backstop ──────────────────────────────────────────────────
    # Deterministic check on the opening sentence of the farmer-facing
    # instruction text — same rationale as _validate_tool() above: the
    # prompt (SYSTEM_REPAIR) already instructs the model to avoid
    # unintroduced jargon, but compliance isn't guaranteed every call.
    # Single targeted reword retry on violation, never a full re-ask.
    # See agent/jargon_guard.py.
    _text_en_before = response.next_step.text_en
    response.next_step.text_en = await apply_jargon_guard(
        response.next_step.text_en, _reword_call_llm, label="text_en"
    )
    # Keep `text` in sync if it was a straight mirror of text_en (the
    # normal case, per _parse_response above) so the reword isn't lost.
    if response.next_step.text == _text_en_before:
        response.next_step.text = response.next_step.text_en

    # Backend enforces structural fields
    response.next_step.required_part = current_step.required_part or ""
    response.next_step.tracking_scope = getattr(current_step, "tracking_scope", "component")
    response.next_step.area_hint = current_step.area_hint
    response.next_step.ar_model = "none"

    # VALIDATE AND NORMALIZE INTERACTION
    response.next_step.interaction = _validate_and_normalize_interaction(
        response.next_step.interaction, 
        response.next_step.required_part,
        current_step.step_type,
        response.next_step.tracking_scope, # Pass explicit scope
    )

    # PREVENT STATUS HALLUCINATION: If we are asking the user a question/action, we MUST be in continue state
    if response.next_step.interaction.type in ("camera", "choice", "boolean", "number"):
        response.status = "continue"

    session.verified_parts.update(response.updated_memory.verified_parts)
    for step in response.updated_memory.diagnostic_path:
        if step not in session.diagnostic_path:
            session.diagnostic_path.append(step)
            
    session.current_stage  += 1
    session.attempt_count  += 1
    session.last_verification = last_verification

    logger.info(
        f"🤖 Agent [{session.machine_type}] stage={session.current_stage} "
        f"status={response.status} part={response.next_step.required_part} "
        f"interaction_type={response.next_step.interaction.type if response.next_step.interaction else 'none'}"
    )
    return response

# Note: Safely delete the `_build_interaction()` helper function and the `_MASTER_AGENT_PROMPT` string entirely.
# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _apply_verification(session: RepairSession, verification: dict) -> None:
    status = verification.get("status", "")
    if status in ("initial", "session_start", "manual_advance", ""):
        return
    part = (
        verification.get("required_part")
        or verification.get("correct_part")
        or verification.get("part_detected")
    )
    if not part or part in ("none", "machine_part", "unknown"):
        return
    status = verification.get("status", "unclear")
    conf   = float(verification.get("confidence", 0))

    if status in ("pass", "verified") and conf >= 0.6:
        session.verified_parts[part] = "ok"
        logger.info(f"✅ [{session.machine_type}] OK: {part} (conf={conf:.2f})")
    elif status == "fail" and conf >= 0.6:
        session.verified_parts[part] = "damaged"
        logger.warning(f"⚠️  [{session.machine_type}] DAMAGED: {part} (conf={conf:.2f})")
    else:
        session.verified_parts[part] = "unclear"
        logger.info(f"❓ [{session.machine_type}] unclear: {part} (conf={conf:.2f})")

    obs = (verification.get("ai_observation") or "").strip()
    if obs and obs.lower() not in ("", "none", "null"):
        session.verified_observations[part] = obs
        logger.info(f"👁️  [{session.machine_type}] observation stored: {part} → {obs[:80]}")


def _build_safety_context(session: RepairSession) -> str:
    """Compact safety summary — accuracy-critical, kept in full."""
    machine_type = session.machine_type
    fuel_parts   = set(get_fuel_system_parts(machine_type))
    is_electric  = is_electric_machine(machine_type)

    damaged = [p for p, s in session.verified_parts.items() if s == "damaged"]
    ok      = [p for p, s in session.verified_parts.items() if s == "ok"]
    lines   = []

    if damaged:
        lines.append(f"DAMAGED: {', '.join(damaged)}")
        fuel_dmg = set(damaged) & fuel_parts
        if fuel_dmg:
            lines.append(f"FUEL_LEAK({', '.join(fuel_dmg)}): block ignition/crank steps.")
    if ok:
        lines.append(f"SKIP(already_ok): {', '.join(ok)}")
    if is_electric:
        lines.append("ELECTRIC: power_off required before every step.")
    if not lines:
        lines.append("No parts verified yet — start with safest external check.")
    return "\n".join(lines)


async def _reword_call_llm(prompt: str) -> str:
    """Plain-text (non-JSON, no system block) LLM call used only by the
    jargon guard's single-field targeted reword retry. Deliberately
    lighter than _call_gemini — we want one plain rewritten sentence
    back, not a structured agent response.
    """
    response = await asyncio.to_thread(
        lambda: groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )
    )
    return response.choices[0].message.content


async def _call_gemini(prompt: str) -> str:
    response = await asyncio.to_thread(
        lambda: groq_chat_completion(
            messages=[
                {"role": "system", "content": REPAIR_SYSTEM_BLOCK},
                {"role": "user", "content": prompt},
            ],
            **JSON_CONFIG,
        )
    )
    return response.choices[0].message.content

def _validate_tool(tool: str | None, machine_type: str) -> str | None:
    """Reject any tool not on the allowed list for this machine. Returns None if invalid."""
    if not tool:
        return None
    allowed = _allowed_tools(machine_type)
    tool_clean = tool.lower().strip().replace(" ", "_")
    if tool_clean in allowed:
        return tool_clean
    # Partial-match fallback (e.g. "adjustable wrench" → "wrench_adjustable")
    for a in allowed:
        if tool_clean in a or a in tool_clean:
            logger.info(f"🔧 [{machine_type}] tool fuzzy-match: '{tool}' → '{a}'")
            return a
    logger.warning(f"⚠️  [{machine_type}] Hallucinated tool rejected: '{tool}'")
    return None

def _parse_interaction(raw: dict | None) -> Interaction | None:
    """Parse interaction block using Pydantic validation. Returns None on any error."""
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return Interaction.model_validate(raw)
    except Exception:
        return None

def _parse_response(raw: str, machine_type: str) -> AgentNextResponse:
    allowed = get_allowed_area_ids(machine_type)

    try:
        data = repair_json(raw)
    except json.JSONDecodeError as exc:
        logger.error(f"❌ [{machine_type}] Invalid JSON: {exc}\n{raw[:400]}")
        return _fallback_response(machine_type, f"JSON parse error: {exc}")

    try:
        ns   = data["next_step"]
        um   = data.get("updated_memory", {})
        area = ns.get("area_hint") or None  # backend will overwrite anyway

        text_en = ns.get("text_en", "")
        if len(text_en.split()) < 25:
            logger.warning(
                f"⚠️  [{machine_type}] Agent text_en only {len(text_en.split())} words — "
                "expected 3–4 guided sentences"
            )
        if not ns.get("text") and text_en:
            ns["text"] = text_en

        # ── Validate required_tool against allowed list ───────────────────
        raw_tool     = ns.get("required_tool")
        validated_tool = _validate_tool(raw_tool, machine_type)

        # ── Warn on missing structured fields ─────────────────────────────
        for field in ("expected_result", "if_failed", "escalate_if"):
            if not ns.get(field):
                logger.warning(f"⚠️  [{machine_type}] Missing structured field: {field}")

        # Extract fields safely
        raw_scope = ns.get("tracking_scope", "component")
        tracking_scope = raw_scope if raw_scope in ("component", "assembly") else "component"
        req_part = ns.get("required_part")

        # LLM Contradiction Guard
        if tracking_scope == "assembly":
            req_part = ""  # Force empty for assemblies
        elif tracking_scope == "component" and not req_part:
            # LLM asked for a component but didn't name it. Safely downgrade to assembly inspection.
            tracking_scope = "assembly"

        return AgentNextResponse(
            status             = data.get("status", "continue"),
            reasoning_summary  = data.get("reasoning_summary", ""),
            next_step = NextStepDetail(
                text             = ns.get("text", ""),
                text_en          = ns.get("text_en", ""),
                text_hi          = ns.get("text_hi", ""),
                visual_cue       = ns.get("visual_cue", ""), # Preserved!
                ar_model         = "none",                   # No fabricated .obj files
                required_part    = req_part,
                tracking_scope   = tracking_scope,
                area_hint        = ns.get("area_hint", "engine_compartment"),
                safety_warning   = ns.get("safety_warning"),
                expected_result    = ns.get("expected_result", ""),
                expected_result_hi = ns.get("expected_result_hi", ""),
                if_failed          = ns.get("if_failed", ""),
                if_failed_hi       = ns.get("if_failed_hi", ""),
                escalate_if        = ns.get("escalate_if", ""),
                escalate_if_hi     = ns.get("escalate_if_hi", ""),
                required_tool      = validated_tool,
                interaction       = _parse_interaction(ns.get("interaction")),
            ),
            updated_memory = UpdatedMemory(
                verified_parts  = um.get("verified_parts", {}),
                diagnostic_path = um.get("diagnostic_path", []),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.error(f"❌ [{machine_type}] Schema error: {exc}")
        return _fallback_response(machine_type, f"Schema error: {exc}")


def _fallback_response(machine_type: str, reason: str) -> AgentNextResponse:
    allowed = get_allowed_area_ids(machine_type)
    return AgentNextResponse(
        status = "escalate",
        reasoning_summary = f"Agent error [{machine_type}]: {reason}",
        next_step = NextStepDetail(
            text="Unable to determine next step. Please consult a mechanic.",
            text_en="Unable to determine next step. Consult a certified mechanic.",
            text_hi="अगला कदम निर्धारित नहीं हो सका। प्रमाणित मैकेनिक से संपर्क करें।",
            visual_cue="none", ar_model="none.obj", required_part="none",
            area_hint=allowed[0] if allowed else "engine_compartment",
            safety_warning="Stop repairs and seek professional assistance.",
            expected_result="N/A",
            expected_result_hi="N/A",
            if_failed="Contact a certified mechanic.",
            if_failed_hi="प्रमाणित मैकेनिक से संपर्क करें।",
            escalate_if="Immediately — agent could not generate a safe step.",
            escalate_if_hi="तुरंत — एजेंट सुरक्षित कदम नहीं बना सका।",
            required_tool=None,
        ),
        updated_memory=UpdatedMemory(verified_parts={}, diagnostic_path=["agent_error"]),
    )

def _validate_and_normalize_interaction(
    interaction: Interaction | None,
    required_part: str | None,
    step_type: str = "inspection",
    tracking_scope: str = "component",
) -> Interaction:
    """Enforces structural rules on LLM interaction schemas.

    Trusts the LLM's own question/options wherever it provided them — per
    SYSTEM_REPAIR, options must stay dynamic and step-specific, generated
    fresh per step (e.g. "Cable looks fine" / "Cable is frayed"), never a
    fixed template. This function only:
      1. Supplies a step_type-aware default when the LLM returns no
         interaction block at all — never blanket-defaults to boolean,
         since that silently erases the camera/choice/number cases too.
      2. Downgrades a camera interaction that has no part to point the
         camera at (structurally impossible to render).
      3. Fills in a GENERIC fallback pair only when the LLM's own
         choice/boolean options are missing or insufficient — it never
         overwrites options the LLM did provide.
    """

    # 1. No interaction block at all — step_type-aware default.
    #    Inspection steps with a required_part ALWAYS default to camera
    #    because the diagnosis prompt now guarantees camera-verifiable steps.
    if not interaction:
        if step_type in ("inspection", "repair") and required_part not in (None, "", "unknown", "none", "machine_part"):
            return Interaction(type="camera", question="", options=[])
        if step_type == "safety":
            return Interaction(
                type="boolean",
                question="Is this safety step complete?",
                options=[
                    InteractionOption(id="yes", label="Yes, done", next_state="continue"),
                    InteractionOption(id="no", label="Not yet", next_state="retry"),
                ]
            )
        if step_type == "verification":
            return Interaction(
                type="choice",
                question="Did the repair fix the problem?",
                options=[
                    InteractionOption(id="fixed", label="Yes, working now", next_state="continue"),
                    InteractionOption(id="not_fixed", label="Still not working", next_state="continue"),
                ]
            )
        return Interaction(
            type="boolean",
            question="",
            options=[InteractionOption(id="yes", label="Done", next_state="continue")],
        )

    # Override: inspection steps should always be camera, never number,
    # boolean, or choice — the LLM doesn't always follow the INTERACTION
    # TYPE priority order in SYSTEM_REPAIR (camera checked first), so this
    # is the deterministic backstop. Without "choice" included here, an
    # inspection step with a real required_part could ship as "choice" and
    # silently skip the entire camera → /verify_step → /inspect_part flow:
    # Flutter's onCapture() branches purely on interaction.type == camera,
    # so anything else just opens the choice/inspection panel instead of
    # running analysis on what the farmer points the camera at.
    #
    # Runs as a pre-pass (mutates type, then falls into the dispatch below)
    # rather than as its own elif branch — a step that DOESN'T qualify for
    # the override (e.g. a legitimate choice step with no required_part,
    # like "what do you smell?") must still reach its normal validation
    # (rule 3/4 below), not silently skip it.
    if (
        interaction.type in ("number", "boolean", "choice")
        and step_type == "inspection"
        and (tracking_scope == "assembly" or required_part)
    ):
        interaction.type = "camera"
        interaction.question = ""
        interaction.options = []

    # 2. Camera interaction validation based on explicit scope
    if interaction.type == "camera":
        if tracking_scope == "assembly":
            interaction.question = ""
            interaction.options = []
        elif tracking_scope == "component" and not required_part:
            # Fallback if somehow a component scope made it here without a part
            interaction.type = "boolean"
            interaction.question = interaction.question or "Action complete?"
            if not interaction.options:
                interaction.options = [InteractionOption(id="yes", label="Done", next_state="continue")]
        else:
            interaction.question = ""
            interaction.options = []

    # 3. Boolean: trust the LLM's own options...
    elif interaction.type == "boolean":
        if not interaction.options:
            logger.warning("Boolean interaction missing options — using dynamic fallback")
            interaction.options = [
                InteractionOption(id="done", label="I did it", next_state="continue"),
                InteractionOption(id="stuck", label="I can't find it", next_state="retry"),
            ]

    # 4. Choice: needs at least 2 usable options.
    elif interaction.type == "choice":
        opts = interaction.options or []
        if len(opts) < 2:
            logger.warning(
                "Downgrading choice interaction: fewer than 2 options (got %d)", len(opts)
            )
            interaction.type = "boolean"
            if len(opts) == 1:
                interaction.options = opts + [InteractionOption(id="stuck", label="I'm stuck / Not done", next_state="retry")]
            else:
                interaction.options = [
                    InteractionOption(id="done", label="I did it", next_state="continue"),
                    InteractionOption(id="stuck", label="I can't find it", next_state="retry"),
                ]

    # 5. Number: FORBIDDEN. Flutter has no numeric input field.
    elif interaction.type == "number":
        logger.warning("Downgrading unsupported 'number' interaction to 'choice'")
        interaction.type = "choice"
        interaction.question = interaction.question or "What is the measurement?"
        interaction.options = [
            InteractionOption(id="opt1", label="Looks correct", next_state="continue"),
            InteractionOption(id="opt2", label="Needs adjustment", next_state="retry"),
        ]

    # 6. None: purely informational step — nothing to render or confirm,
    #    the agent auto-advances once the farmer reads it.
    elif interaction.type == "none":
        interaction.question = ""
        interaction.options = []

    # 7. Text: Interaction.type's Pydantic Literal in models.py permits
    #    "text", but Flutter's InteractionType has no case for it anywhere
    #    in ar_controller.dart or scanning_indicator.dart — nothing renders
    #    it. SYSTEM_REPAIR never instructs the LLM to emit it, so this
    #    should be unreachable in practice, but downgrade defensively
    #    rather than ship an interaction the phone can't display.
    elif interaction.type == "text":
        logger.warning("Downgrading unsupported 'text' interaction to boolean — no Flutter renderer exists for it")
        interaction.type = "boolean"
        interaction.options = interaction.options or [
            InteractionOption(id="yes", label="Done", next_state="continue")
        ]

    return interaction