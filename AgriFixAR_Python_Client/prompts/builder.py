"""
prompts/builder.py
Assembles chat messages. Pre-built system blocks avoid repeated string concat.
"""

from __future__ import annotations
from typing import List, Dict

from prompts.sections.base import SYSTEM_BASE, GROUNDING, JSON_RULE
from prompts.sections.repair import SYSTEM_REPAIR

# Pre-built system blocks (computed once)
REPAIR_SYSTEM_BLOCK = "\n\n".join([SYSTEM_BASE, SYSTEM_REPAIR, GROUNDING, JSON_RULE])


class PromptBuilder:
    """Builds chat messages. Use pre-built blocks for common patterns."""

    def __init__(self):
        self._system_parts: list[str] = []
        self._user_parts: list[str] = []

    def add_system(self, section: str) -> "PromptBuilder":
        if section.strip():
            self._system_parts.append(section.strip())
        return self

    def add_user(self, section: str) -> "PromptBuilder":
        if section.strip():
            self._user_parts.append(section.strip())
        return self

    def build_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self._system_parts:
            messages.append({"role": "system", "content": "\n\n".join(self._system_parts)})
        if self._user_parts:
            messages.append({"role": "user", "content": "\n\n".join(self._user_parts)})
        return messages