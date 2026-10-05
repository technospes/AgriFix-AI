# agent/validation.py
#
# Centralized repair-plan structural validation.
#
# Called from exactly two places:
#   1. diagnosis_service.py — right after the final step list is settled
#      (dedup + procedure_validator injection both done). This is the
#      plan's point of origin; if it's invalid here, that's where it
#      must be caught.
#   2. repair_agent.py — defense-in-depth, immediately before the agent
#      trusts a plan handed to it by a session. Should never actually
#      fire if (1) is doing its job, but a stale cached plan or a future
#      code path that bypasses diagnosis_service could still reach the
#      agent, so this stays as a second gate.
#
# Anywhere else that needs to check plan integrity should call
# validate_repair_plan_steps(), not reimplement these checks.
from __future__ import annotations
from typing import Any, Sequence


class InvalidRepairPlan(Exception):
    """A repair plan is structurally broken.

    This is a BACKEND DEFECT, not a mechanical fault with the farmer's
    machine — nothing is wrong with the tractor, something is wrong with
    the service. Callers at the API boundary (main.py's /diagnose* and
    /agent/* endpoints) must catch this separately from normal escalation
    handling and return a generic service-error response (e.g. HTTP 500,
    or a "something went wrong, please try again" message) — never the
    "consult a certified mechanic" escalation card, which would mislead
    the farmer into thinking their equipment is the problem.
    """


def _get(obj: Any, key: str) -> Any:
    """Attribute-or-dict-key access — steps are plain dicts in
    diagnosis_service.py (raw JSON, pre-Pydantic) but RepairPlanStep
    model instances in repair_agent.py. This lets one validator serve
    both without either caller needing to convert first."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def validate_repair_plan_steps(steps: Sequence[Any], *, context: str = "") -> None:
    """Raise InvalidRepairPlan if `steps` is structurally broken.

    Checks (in order, so the first failure reported is the most useful):
      - plan is non-empty
      - every step has a non-blank step_id
      - all step_ids are unique
      - every step has an action and an area_hint (minimum viable content
        for the agent to build a prompt / for Flutter to render anything)

    `context` is a short free-text tag (e.g. "machine=tractor") included
    in the raised message purely to make server logs greppable — it has
    no effect on the checks themselves.
    """
    suffix = f" ({context})" if context else ""

    if not steps:
        raise InvalidRepairPlan(f"Empty repair plan{suffix}")

    ids = [_get(s, "step_id") for s in steps]

    blank = [i for i in ids if not i]
    if blank:
        raise InvalidRepairPlan(
            f"Blank step_id in repair plan{suffix}: {ids!r}"
        )

    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise InvalidRepairPlan(
            f"Duplicate step_id in repair plan{suffix}: {ids!r} "
            f"(duplicated: {sorted(dupes)!r})"
        )

    for s in steps:
        step_id = _get(s, "step_id") or "?"
        if not _get(s, "action"):
            raise InvalidRepairPlan(
                f"Missing action for step {step_id!r}{suffix}"
            )
        if not _get(s, "area_hint"):
            raise InvalidRepairPlan(
                f"Missing area_hint for step {step_id!r}{suffix}"
            )