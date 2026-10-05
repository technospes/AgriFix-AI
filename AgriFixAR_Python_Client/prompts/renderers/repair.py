"""
prompts/renderers/repair.py
Renders the repair agent prompt from a PromptContext snapshot.
"""

from prompts.context import PromptContext
from prompts.sections.repair_schema import REPAIR_JSON_SCHEMA

REPAIR_PROMPT_VERSION = "v3.2"

_TASK_TEMPLATE = """\
Explain one step of a diagnosis plan to a first-time farmer.

MACHINE: {machine_type}
STEP: {action}
DESCRIPTION: {description}
PART: {required_part}
AREA: {area_hint}
LOCATION: {area_description}
LANDMARKS: {area_landmarks}
TYPE: {step_type}
ATTEMPT: {attempt_count}

VERIFIED: {verified_parts_json}
CAMERA: {visual_observations}
LAST RESULT: {last_verification_json}
SAFETY: {safety_context}
AREAS: {relevant_areas}
PARTS: {relevant_parts}
{tools_block}

{json_schema}"""


def render_repair_prompt(ctx: PromptContext) -> str:
    """Render the repair agent task prompt from a PromptContext."""
    return _TASK_TEMPLATE.format(
        machine_type=ctx.machine_type,
        action=ctx.action,
        description=ctx.description,
        required_part=ctx.required_part,
        area_hint=ctx.area_hint,
        area_description=ctx.area_description,
        area_landmarks=ctx.area_landmarks,
        step_type=ctx.step_type,
        attempt_count=ctx.attempt_count,
        verified_parts_json=ctx.verified_parts_json,
        visual_observations=ctx.visual_observations,
        last_verification_json=ctx.last_verification_json,
        safety_context=ctx.safety_context,
        relevant_areas=ctx.relevant_areas,
        relevant_parts=ctx.relevant_parts,
        tools_block=ctx.tools_block,
        json_schema=REPAIR_JSON_SCHEMA,
    )