"""
agent/models.py
Pydantic data structures for the AgriFix repair agent system.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Literal
from pydantic import BaseModel, Field
from enum import Enum, StrEnum

class VerificationMode(StrEnum):
    CAMERA = "camera"
    CONFIRMATION = "confirmation"
    MEASUREMENT = "measurement"
    AUDIO = "audio"


class Verification(BaseModel):
    mode: VerificationMode = VerificationMode.CONFIRMATION
    advance: str = "automatic"  # "automatic" | "agent"

class StepType(str, Enum):
    SAFETY = "safety"
    INSPECTION = "inspection"
    REPAIR = "repair"
    VERIFICATION = "verification"

# ─────────────────────────────────────────────
# Session state stored in memory
# ─────────────────────────────────────────────
class InteractionOption(BaseModel):
    """One choice the farmer can select."""
    id: str                                    # stable ID e.g. "normal", "bulging"
    label: str                                 # display text e.g. "Everything looks normal"
    next_state: str = ""                       # semantic signal e.g. "continue", "oil_leak_detected"


class Interaction(BaseModel):
    """How the farmer should respond to this step. Flutter renders this directly."""
    type: Literal["choice", "camera", "boolean", "text", "number", "none"] = "none"
    question: str = ""
    options: List[InteractionOption] = Field(default_factory=list)
    required: bool = True

class RepairPlanStep(BaseModel):
    step_id: str
    action: str
    description: str = ""
    required_part: Optional[str] = None
    tracking_scope: Literal["component", "assembly"] = "component"
    area_hint: str = "engine_compartment"
    step_type: StepType = StepType.INSPECTION
    verification: Optional[Verification] = None
    requires_disassembly: bool = False


class RepairPlan(BaseModel):
    """Immutable diagnosis result — stored as-is in the session."""
    machine_type: str
    confidence: float = 0.0
    likely_fault: str = ""
    rag_score: float = 0.0
    steps: List[RepairPlanStep] = Field(default_factory=list)


class VerifiedPart(BaseModel):
    """Rich verification record per part."""
    status: Literal["ok", "damaged", "unclear"] = "unclear"
    confidence: float = 0.0
    source: Literal["vision", "user", "agent"] = "user"
    timestamp: str = ""
    notes: str = ""

class RepairSession(BaseModel):
    session_id: str
    machine_type: str
    problem: str
    verified_parts: Dict[str, Literal["ok", "damaged", "unclear"]] = Field(default_factory=dict)
    verified_observations: Dict[str, str] = Field(default_factory=dict)
    # ↑ Gemini's actual visual finding per part, e.g.:
    #   {"battery_terminal": "white powder visible on both clamps — corrosion confirmed"}
    diagnostic_path: List[str] = Field(default_factory=list)
    generated_steps: List[str] = Field(default_factory=list)
    # ── Immutable diagnosis plan ──────────────────────────────────────────────
    repair_plan: Optional[RepairPlan] = None
    current_step_id: str = ""
    verified_parts_rich: Dict[str, VerifiedPart] = Field(default_factory=dict)
    current_stage: int = 0
    attempt_count: int = 0
    last_verification: Optional[Dict] = None
    language: str = "en"
    # Parts/areas whose cover/housing has already been opened this session
    # (farmer answered the access-confirmation boolean with "opened").
    # Populated in repair_agent.py's decide_next_step() the moment a
    # requires_disassembly step resolves with "continue"/answer_bool=True —
    # same place _apply_verification() records verified_parts.
    #
    # Two kinds of entries share this one flat list, disambiguated by prefix:
    #   - "<part_id>"        e.g. "capacitor"            — that exact part's
    #                          own access step was confirmed.
    #   - "area:<area_hint>" e.g. "area:motor_housing"    — the housing/cover
    #                          for that whole area was confirmed open, so
    #                          EVERY part located in that area_hint counts
    #                          as reachable, not just the part named on the
    #                          access step itself.
    # The area entry is what actually matters in practice: diagnosis_service
    # never guarantees the access step's required_part matches every later
    # inspection step's required_part under the same cover (e.g. access
    # step names "motor_cover", later steps inspect "capacitor" then
    # "start_relay") — part-only tracking silently stops working after the
    # first step. Read by repair_agent.py's decide_next_step() (deterministic
    # override of requires_disassembly) and by _build_safety_context() (the
    # ACCESS_OPENED / ACCESS_OPENED_AREA lines the LLM's prompt reads).
    access_achieved: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# /agent/next  —  request / response
# ─────────────────────────────────────────────

class AgentNextRequest(BaseModel):
    session_id: str
    last_verification_result: Dict  # full JSON blob from /verify_step


class NextStepDetail(BaseModel):
    # ── Core instruction (bilingual) ─────────────────────────────────────────
    text: str
    text_en: str
    text_hi: str

    # ── AR / visual anchoring ────────────────────────────────────────────────
    visual_cue: str
    ar_model: str = "none"
    required_part: Optional[str] = None
    tracking_scope: Literal["component", "assembly"] = Field(default="component")
    area_hint: str
    requires_disassembly: bool = False

    # ── Safety ──────────────────────────────────────────────────────────────
    safety_warning: Optional[str] = None

    # ── Structured repair output fields ──────────────────────────────────────
    # What the farmer should observe when the step succeeds (physical)
    expected_result: str = ""
    expected_result_hi: str = ""

    # Most likely cause if the step fails + single corrective action
    if_failed: str = ""
    if_failed_hi: str = ""

    # Concrete condition that means STOP and call a mechanic
    escalate_if: str = ""
    escalate_if_hi: str = ""

    # Tool from the machine-specific allowed list, or null for visual-only steps
    required_tool: Optional[str] = None
    # ── Interactive feedback ──────────────────────────────────────────────────
    # Backend-driven: Flutter renders buttons, camera, yes/no, text, or numeric
    # input based on this. null means informational step with no response needed.
    interaction: Optional[Interaction] = None


class UpdatedMemory(BaseModel):
    verified_parts: Dict[
        str,
        Literal["ok", "damaged", "unclear"]
    ]
    diagnostic_path: List[str]


class AgentNextResponse(BaseModel):
    status: Literal["continue", "resolved", "escalate", "unsafe"]
    reasoning_summary: str
    next_step: NextStepDetail
    updated_memory: UpdatedMemory


# ─────────────────────────────────────────────
# /agent/session  —  session creation
# ─────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    machine_type: str
    problem_description: str
    language: str = "en"
    diagnosis_steps: List[Dict] = Field(default_factory=list)
    # ↑ Optional: pass the steps array from /diagnose response so the agent
    #   knows exactly which parts the plan covers and in what order.


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str