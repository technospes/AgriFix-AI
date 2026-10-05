"""
agent/jargon_guard.py

Deterministic backstop for farmer-facing jargon — same architectural role
as _validate_tool() in repair_agent.py and the guards in safety_rules.py:
runs AFTER the LLM responds, cannot be bypassed by prompt injection, and
degrades gracefully rather than trusting the model to have followed a
prose instruction buried in a long system prompt.

Used by both pipelines that produce farmer-facing text:
  - diagnosis_service.py  (_post_process_diagnosis — initial plan steps)
  - repair_agent.py       (decide_next_step — per-step agent responses)

Nothing here is part-specific or machine-specific. JARGON_TERMS is a
closed linguistic category (generic mechanical-component nouns), the
same tier of resource as a stopword list — extend it when a new category
of term shows up in violations, never per-part or per-machine.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Generic mechanical-jargon terms a first-time, zero-literacy farmer
# wouldn't know. Single source of truth — both prompts already instruct
# the model to avoid these; this is the deterministic check behind that
# instruction, not a replacement for it.
JARGON_TERMS = {
    "shaft", "coupling", "bearing", "terminal board", "capacitor",
    "gland", "seal", "impeller", "solenoid", "armature", "rotor",
    "stator", "bushing", "gasket", "manifold", "actuator", "relay",
    "diaphragm", "flange", "spindle", "sprocket",
}

# Max characters sent back to the model on a targeted reword call — keeps
# the retry cheap; we're only ever fixing one sentence, not a paragraph.
_REWORD_MAX_CHARS = 600


def find_jargon_violation(text: str) -> Optional[str]:
    """Pure string check, no LLM call, ~0 cost.

    Returns the offending term if the FIRST SENTENCE of `text` contains
    unintroduced jargon, else None. Only checks the opening sentence —
    by design, the technical name is expected (and fine) later in the
    text, once the part has been visually introduced.
    """
    if not text:
        return None
    first_sentence = text.split(".")[0].lower()
    for term in JARGON_TERMS:
        if term in first_sentence:
            return term
    return None


async def reword_without_jargon(
    text: str,
    violation: str,
    call_llm,
    *,
    max_attempts: int = 2,
) -> str:
    """Single targeted re-ask for just the offending field — not a full
    plan/step regeneration. `call_llm` is an async callable(prompt) -> str,
    passed in by the caller so this module stays decoupled from any one
    pipeline's LLM client (diagnosis_service.py and repair_agent.py each
    already have their own).

    Bounded retry (default 2 attempts): a reword can fix the original
    term but introduce a *different* jargon word in the process (e.g.
    "shaft" removed, but the rewrite leans on "coupling" instead) — so
    every attempt is re-checked for ANY violation, not just whether the
    original term came back. On exhausting attempts, or any failure,
    returns the original text unchanged — never raises, never blocks
    the response. A missed reword is a quality miss, not an outage.
    """
    current_text = text
    current_violation = violation
    for attempt in range(1, max_attempts + 1):
        prompt = (
            "Rewrite the following farmer-facing sentence so it does NOT use "
            f"the word '{current_violation}' anywhere in the opening sentence, "
            "and does not introduce any OTHER technical part name in its "
            "place (e.g. shaft, coupling, bearing, gland, seal, impeller, "
            "capacitor, terminal board). "
            "Describe the part by its appearance, color, shape, size, or "
            "position next to a permanent landmark instead. Do not invent "
            "any detail that isn't already implied by the original text. "
            "You may introduce the technical name later in the text, just "
            "not as the first thing mentioned. Return ONLY the rewritten "
            "text, no preamble, no quotes.\n\n"
            f"ORIGINAL:\n{current_text[:_REWORD_MAX_CHARS]}"
        )
        try:
            reworded = await call_llm(prompt)
            reworded = reworded.strip().strip('"')
        except Exception as exc:
            logger.warning("jargon_guard: reword attempt %d failed (%s) — keeping original", attempt, exc)
            return text

        if not reworded:
            continue

        still_violating = find_jargon_violation(reworded)
        if still_violating is None:
            return reworded

        logger.warning(
            "jargon_guard: reword attempt %d left '%s' in first sentence "
            "(started from '%s')", attempt, still_violating, violation
        )
        current_text, current_violation = reworded, still_violating

    logger.warning(
        "jargon_guard: gave up after %d attempts, term '%s' persisted — keeping original text",
        max_attempts, violation,
    )
    return text


async def apply_jargon_guard(
    text: str,
    call_llm,
    *,
    label: str = "",
) -> str:
    """Convenience wrapper: check, and if violated, attempt one reword.
    Always returns usable text; never raises.
    """
    violation = find_jargon_violation(text)
    if not violation:
        return text
    logger.info("jargon_guard: '%s' found in first sentence%s — attempting reword",
                violation, f" [{label}]" if label else "")
    return await reword_without_jargon(text, violation, call_llm)