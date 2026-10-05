from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Validation rules (based on IEC 60204-1 and ISO 11684 safety principles) ───

_SHUTDOWN_KEYWORDS = {
    "switch off", "turn off", "disconnect power", "isolate power",
    "stop engine", "power off", "mcb off", "breaker off",
    "depressurise", "drain", "shut down", "remove ignition key",
    "bijli band", "switch band", "engine band",
}

_OPEN_PANEL_KEYWORDS = {
    "open panel", "remove cover", "open terminal box", "open casing",
}

_LOCKOUT_KEYWORDS = {
    "lock out", "lockout", "tagout", "tag out", "lock the breaker",
    "place warning tag", "padlock", "do not operate tag",
}

_ELECTRICAL_WORK_KEYWORDS = {
    "capacitor", "terminal", "winding", "relay", "mcb", "fuse",
    "voltage", "wire", "battery", "alternator", "motor terminal",
    "circuit", "live", "current",
}

_HYDRAULIC_WORK_KEYWORDS = {
    "hydraulic", "cylinder", "pressure", "oil pressure", "valve",
    "hose", "fitting", "ram", "accumulator",
}

_ROTATING_PARTS_KEYWORDS = {
    "belt", "pulley", "shaft", "impeller", "blade", "pto", "fan",
    "flywheel", "gear", "coupling", "rotating",
}

_ELECTRICAL_TOOLS = {"multimeter", "insulated screwdriver", "gloves", "tester"}
_MECHANICAL_TOOLS = {"spanner", "wrench", "pliers", "screwdriver"}


# ── Output dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str          # "CRITICAL" | "HIGH" | "WARNING" | "INFO"
    rule:     str          # rule name
    detail:   str          # human-readable explanation
    auto_fix: bool = False # True if automatically corrected


@dataclass
class ProcedureValidation:
    passed:         bool
    issues:         List[ValidationIssue]
    safe_steps:     List[Dict[str, Any]]
    original_steps: List[Dict[str, Any]]
    risk_level:     str
    machine_type:   str
    shutdown_ok:    bool = False
    lockout_ok:     bool = False
    tools_ok:       bool = False

    def critical_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "CRITICAL"]

    def high_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "HIGH"]

    def summary(self) -> str:
        n_crit = len(self.critical_issues())
        n_high = len(self.high_issues())
        n_warn = len([i for i in self.issues if i.severity == "WARNING"])
        return (
            f"ProcedureValidation: passed={self.passed} | "
            f"CRITICAL={n_crit} HIGH={n_high} WARN={n_warn} | "
            f"shutdown_ok={self.shutdown_ok} lockout_ok={self.lockout_ok} tools_ok={self.tools_ok}"
        )


# ── Text helpers ───────────────────────────────────────────────────────────────

def _step_text(step: Dict[str, Any]) -> str:
    parts = [
        str(step.get("instruction", "")),
        str(step.get("text_en", "")),
        str(step.get("text_hi", "")),
        str(step.get("description", "")),
        str(step.get("action", "")),
        str(step.get("warning", "")),
    ]
    return " ".join(p for p in parts if p).lower()

def _all_steps_text(steps: List[Dict[str, Any]]) -> str:
    return " ".join(_step_text(s) for s in steps)

def _keywords_present(text: str, keywords: set) -> bool:
    return any(kw in text for kw in keywords)

def _detect_work_type(steps_text: str) -> dict:
    return {
        "electrical": _keywords_present(steps_text, _ELECTRICAL_WORK_KEYWORDS),
        "hydraulic":  _keywords_present(steps_text, _HYDRAULIC_WORK_KEYWORDS),
        "rotating":   _keywords_present(steps_text, _ROTATING_PARTS_KEYWORDS),
    }


# ── Safety step injectors ──────────────────────────────────────────────────────

def _make_shutdown_step(machine_type: str, work_type: dict) -> Dict[str, Any]:
    from utils.machine_registry import get_shutdown_instruction
    shutdown = get_shutdown_instruction(machine_type)
    
    return {
        "step_number": 1,
        "text_en": shutdown["instruction_en"],
        "text_hi": shutdown["instruction_hi"],
        "description": shutdown["instruction_en"],
        "instruction": shutdown["instruction_en"],
        "instruction_hi": shutdown["instruction_hi"],
        "visual_cue": f"Locate the {shutdown['required_part'].replace('_', ' ')} and confirm it is in the OFF/safe position.",
        "part": shutdown["required_part"],
        "action": shutdown["action"],
        "required_part": shutdown["required_part"],
        "area_hint": shutdown["area_hint"],
        "is_safety_step": True,
        "_auto_injected": True,
        "_injection_rule": "SHUTDOWN_STEP_INJECTION",
    }

def _make_lockout_step() -> Dict[str, Any]:
    _instruction    = "Apply lockout/tagout: switch off the main isolator, attach a lockout padlock, and affix a DO NOT OPERATE tag before any work begins."
    _instruction_hi = "Lockout/tagout lagayein: main isolator band karein, padlock lagayein, aur 'DO NOT OPERATE' tag lagayein kaam shuru karne se pehle."
    return {
        # ── Flutter-required fields ──────────────────────────────────────────
        "step_number":    1,
        "text_en":        _instruction,
        "text_hi":        _instruction_hi,
        # Legacy aliases
        "instruction":    _instruction,
        "instruction_hi": _instruction_hi,
        "visual_cue":     "Locate the main isolator switch. Confirm padlock is attached and DO NOT OPERATE tag is visible.",
        "part":           "main_isolator",
        "action":         "lockout_tagout",
        "required_part":  "main_isolator",
        "area_hint":      "control_panel",
        # ── Safety metadata ──────────────────────────────────────────────────
        "is_safety_step": True,
        "_auto_injected": True,
        "_injection_rule": "LOCKOUT_TAGOUT_INJECTION",
    }

def _renumber_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for i, step in enumerate(steps, start=1):
        step = dict(step)
        step["step_number"] = i
        steps[i - 1] = step
    return steps


# ── Core validator ─────────────────────────────────────────────────────────────

def validate_procedure(
    steps:        List[Dict[str, Any]],
    machine_type: str,
    risk_level:   str = "MEDIUM",
    tools_list:   Optional[List[str]] = None,
    safe_inject:  bool = True,
) -> ProcedureValidation:
    
    issues: List[ValidationIssue] = []
    safe_steps = [dict(s) for s in steps]

    if not steps:
        logger.info("validate_procedure: empty step list — nothing to validate")
        return ProcedureValidation(
            passed=True, issues=[], safe_steps=[], original_steps=[],
            risk_level=risk_level, machine_type=machine_type,
        )

    steps_text = _all_steps_text(steps)
    step1_text = _step_text(steps[0]) if steps else ""
    work_type  = _detect_work_type(steps_text)
    is_electric = work_type["electrical"] or machine_type in ("electric_motor",)
    is_high_risk = risk_level in ("HIGH", "CRITICAL")

    # ── Duplicate & Order Tracking ──
    seen_instructions = set()
    shutdown_idx = -1
    open_idx = -1

    for i, step in enumerate(steps):
        norm = _step_text(step).strip()
        
        # Duplicate Step Detection
        if norm in seen_instructions:
            issues.append(ValidationIssue("WARNING", "DUPLICATE_STEP", "Duplicate inspection or step detected.", auto_fix=False))
        seen_instructions.add(norm)
        
        # Track indices for unsafe order
        if _keywords_present(norm, _SHUTDOWN_KEYWORDS):
            if shutdown_idx == -1: shutdown_idx = i
        if _keywords_present(norm, _OPEN_PANEL_KEYWORDS):
            if open_idx == -1: open_idx = i
            
        # Pump / Electrical Safety Checks
        if "dry run" in norm and risk_level in ["HIGH", "CRITICAL"]:
            issues.append(ValidationIssue("HIGH", "UNSAFE_ACTION", "Dry run instruction on high-risk machine.", auto_fix=False))
        if "live terminal" in norm or "live wire" in norm:
             issues.append(ValidationIssue("CRITICAL", "UNSAFE_ACTION", "Work on live terminals detected.", auto_fix=False))
        if "rotating shaft" in norm and "touch" in norm:
             issues.append(ValidationIssue("HIGH", "UNSAFE_ACTION", "Dangerous interaction with rotating shaft.", auto_fix=False))

    # Unsafe Order Detection (Open panel before isolate)
    if open_idx != -1 and (shutdown_idx == -1 or open_idx < shutdown_idx):
        issues.append(ValidationIssue("HIGH", "UNSAFE_ORDER", "Panel opened before power isolation.", auto_fix=False))

    # ── Rule 1: Shutdown/isolation step ──
    shutdown_present = _keywords_present(steps_text, _SHUTDOWN_KEYWORDS)
    shutdown_first   = _keywords_present(step1_text, _SHUTDOWN_KEYWORDS)
    # FIX A2: shutdown_ok is set AFTER injection decisions, not before.
    # The old code set shutdown_ok=shutdown_present prematurely, then re-set it
    # after injection — making the CRITICAL issue appear alongside shutdown_ok=True
    # in the log. Now shutdown_ok reflects the final post-injection state only.
    shutdown_ok = False  # computed at the end of Rule 1

    if is_high_risk or is_electric or work_type["hydraulic"]:
        if not shutdown_present:
            issues.append(ValidationIssue(
            severity="WARN" if not safe_inject else "CRITICAL",
            rule="SHUTDOWN_STEP_MISSING",
            detail=f"No shutdown/isolation step found for {machine_type} (risk={risk_level}).",
            auto_fix=safe_inject,
            ))
            if safe_inject:
                safe_steps.insert(0, _make_shutdown_step(machine_type, work_type))
                safe_steps = _renumber_steps(safe_steps)
                # shutdown_ok is True only because we just injected it
                shutdown_ok = True
            else:
                # No injection — shutdown is genuinely absent and unresolved
                shutdown_ok = False
        elif not shutdown_first and is_electric:
            issues.append(ValidationIssue(
                severity="HIGH", rule="SHUTDOWN_NOT_FIRST_STEP",
                detail="Shutdown exists but is not Step 1.",
                auto_fix=False,
            ))
            shutdown_ok = True  # present but in wrong position
        else:
            shutdown_ok = True  # present and correctly placed
    else:
        # Machine/risk profile does not require a shutdown step
        shutdown_ok = True

    # ── Rule 2: Lockout/tagout ──
    lockout_present = _keywords_present(steps_text, _LOCKOUT_KEYWORDS)
    lockout_ok = False  # computed below

    if risk_level == "CRITICAL" and is_electric and work_type["rotating"]:
        if not lockout_present:
            issues.append(ValidationIssue(
                severity="CRITICAL", rule="LOCKOUT_TAGOUT_MISSING",
                detail="Lockout/tagout step is mandatory per IEC 60204-1.",
                auto_fix=safe_inject,
            ))
            if safe_inject:
                safe_steps.insert(0, _make_lockout_step())
                safe_steps = _renumber_steps(safe_steps)
                lockout_ok = True
            else:
                lockout_ok = False
        else:
            lockout_ok = True
    else:
        lockout_ok = True  # not required for this profile

    # ── Rule 3: Electrical PPE ──
    if is_electric and is_high_risk:
        tool_text = " ".join(str(t).lower() for t in (tools_list or []))
        if not _keywords_present(tool_text + steps_text, {"insulated", "gloves", "rubber gloves", "ppe"}):
            issues.append(ValidationIssue(
                severity="WARNING", rule="ELECTRICAL_PPE_NOT_MENTIONED",
                detail="Procedure does not mention insulated tools or gloves.",
                auto_fix=False,
            ))

    # ── Determine overall pass/fail ──
    # FIX A2 (continued): passed=True only when ALL non-auto-fixed CRITICALs and HIGHs
    # are absent. auto_fix=True issues that were successfully injected do NOT
    # prevent passed=True — which is correct, because the injection resolved them.
    # The logging block below fires unconditionally for ALL issues regardless of passed.
    remaining_criticals = [i for i in issues if i.severity == "CRITICAL" and not i.auto_fix]
    remaining_highs     = [i for i in issues if i.severity == "HIGH"     and not i.auto_fix]
    passed = len(remaining_criticals) == 0 and len(remaining_highs) == 0

    validation = ProcedureValidation(
        passed=passed, issues=issues, safe_steps=safe_steps,
        original_steps=steps, risk_level=risk_level,
        machine_type=machine_type, shutdown_ok=shutdown_ok,
        lockout_ok=lockout_ok, tools_ok=True,
    )

    logger.info("Procedure validation: %s", validation.summary())
    for issue in issues:
        level = logging.WARNING if issue.severity in ("CRITICAL","HIGH") else logging.DEBUG
        logger.log(level, "  [%s] %s: %s", issue.severity, issue.rule, issue.detail[:80])

    return validation