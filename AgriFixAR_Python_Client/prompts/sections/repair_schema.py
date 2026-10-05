"""
prompts/sections/repair_schema.py
JSON schema contract for the repair agent — reused, not duplicated.
"""

REPAIR_JSON_SCHEMA = """\
Return:
{{
  "status": "continue" | "escalate" | "unsafe",
  "reasoning_summary": "<1 sentence>",
  "next_step": {{
    "text": "<copy of text_en>",
    "text_en": "string (3-4 sentences)",
    "text_hi": "string",
    "requires_disassembly": "boolean — true only if this PART is enclosed and no prior VERIFIED entry shows its cover/housing already removed this session (see ACCESS-GATE)",
    "safety_warning": "string|null",
    "expected_result": "string",
    "expected_result_hi": "string",
    "if_failed": "string",
    "if_failed_hi": "string",
    "escalate_if": "string",
    "escalate_if_hi": "string",
    "required_tool": "string|null",
    "interaction": {{
      "type": "boolean" | "choice" | "camera" | "number" | "none",
      "question": "string",
      "options": [
        {{"id": "opt_1", "label": "<farmer's own words for THIS step's outcome — never generic Yes/No/Done text>", "next_state": "continue"}}
      ]
    }}
  }},
  "updated_memory": {{
    "verified_parts": {{"<part>": "ok|damaged|unclear"}},
    "diagnostic_path": ["<step>"]
  }}
}}"""