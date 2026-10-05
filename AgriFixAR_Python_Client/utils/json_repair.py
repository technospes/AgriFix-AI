from __future__ import annotations
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def repair_json(raw: str) -> Any:
    """
    Attempt to parse raw string as JSON, applying progressive repair steps.

    Raises json.JSONDecodeError only if all repair strategies fail.
    Callers should wrap in try/except for total safety.
    """
    if not raw or not isinstance(raw, str):
        raise json.JSONDecodeError("Empty or non-string input", raw or "", 0)

    text = _strip_fences(raw)

    # 1. Fast path — already valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Sequential repair attempts — try after each fix
    repairs = [
        _fix_trailing_commas,
        _fix_single_quotes,
        _fix_invalid_escapes,
        _fix_unterminated_string,
    ]

    for repair_func in repairs:
        try:
            text = repair_func(text)
            return json.loads(text)
        except json.JSONDecodeError:
            continue

    # 3. Final fallback: extract JSON object with regex, parse with strict=False
    extracted = _extract_json_object(text)
    if extracted:
        try:
            return json.loads(extracted, strict=False)
        except json.JSONDecodeError:
            pass

    logger.error("JSON repair failed after all strategies. Raw[:200]: %s", raw[:200])
    raise json.JSONDecodeError("All repair strategies failed", raw, 0)


# ── Repair helpers ────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?\s*```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ]."""
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    return text


def _fix_single_quotes(text: str) -> str:
    """
    Replace single-quoted JSON keys/values with double-quoted equivalents.
    Matches '...' patterns that look like JSON tokens (not prose apostrophes).
    """
    # Matches single-quoted strings: 'some text with possible \' escapes'
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)
    return text


def _fix_invalid_escapes(text: str) -> str:
    """
    Fix invalid Unicode escape sequences like \\uXXXX where XXXX is malformed.
    Gemini sometimes outputs \\u0000 or truncated \\u sequences.
    Replace invalid \\u escapes with \\ufffd (Unicode replacement character).
    """
    def _replace_invalid_unicode(match):
        hex_part = match.group(1)
        if len(hex_part) == 4:
            try:
                int(hex_part, 16)
                return match.group(0)  # Valid, keep as-is
            except ValueError:
                pass
        # Invalid or truncated — replace with replacement character
        return "\\ufffd"

    text = re.sub(r'\\u([0-9a-fA-F]{0,4})', _replace_invalid_unicode, text)
    return text


def _fix_unterminated_string(text: str) -> str:
    """
    If JSON ends with an open string (odd number of unescaped "),
    close it and close any open {} or [].
    Handles token-limit truncation: {"key": "partial value → {"key": "partial value"}
    """
    # Count unescaped double-quotes
    unescaped_quotes = len(re.findall(r'(?<!\\)"', text))
    if unescaped_quotes % 2 == 0:
        return text  # Balanced

    text = text.rstrip()
    text += '"'  # Close the open string

    # Close any open brackets
    open_curly = text.count('{') - text.count('}')
    open_square = text.count('[') - text.count(']')
    text += '}' * max(0, open_curly)
    text += ']' * max(0, open_square)

    return text


def _extract_json_object(text: str) -> str | None:
    """
    Last-resort: find the first { ... } block using brace matching.
    Returns the substring or None if no valid brace pair found.
    """
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None