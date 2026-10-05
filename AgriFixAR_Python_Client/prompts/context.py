"""
prompts/context.py
Immutable context object passed to prompt renderers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass(frozen=True)
class PromptContext:
    """Snapshot of current state for prompt rendering. Immutable — never modified after creation."""
    machine_type: str = ""
    machine_label: str = ""
    language: str = "en"

    # Current step (agent only)
    action: str = ""
    description: str = ""
    required_part: str = ""
    area_hint: str = ""
    step_type: str = "inspection"
    attempt_count: int = 0

    # Area context
    area_description: str = ""
    area_landmarks: str = ""

    # Filtered machine data
    relevant_parts: str = ""
    relevant_areas: str = ""
    tools_block: str = ""
    verification_capability: str = ""

    # Session state
    verified_parts_json: str = "{}"
    visual_observations: str = "None"
    last_verification_json: str = "{}"
    safety_context: str = ""

    # Diagnosis-specific
    problem_text: str = ""
    rag_context: str = ""
    allowed_areas: str = ""
    parts_list: str = ""
    safety_keywords: str = ""
    router_symptoms: str = ""
    has_visual_frames: bool = False
    context_quality: str = "strong"
    top_score: float = 0.0