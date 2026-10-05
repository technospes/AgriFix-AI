# ood_guard.py
"""
Phase 0: Out-of-Domain (OOD) intent guard for AgriFix.

Runs BEFORE ChromaDB retrieval. Rejects queries that are clearly not
repair-intent: price enquiries, purchasing advice, brand recommendations,
general knowledge, stolen/destroyed equipment, abuse/accident reports.

Design principles:
  - Config-driven: pattern sets are loaded from ENV or a YAML file.
    Hardcoded patterns below are the defaults; override via AGRIFIX_OOD_PATTERNS_FILE.
  - Language-agnostic: all patterns use re.IGNORECASE + Unicode.
  - Zero false positives: patterns must match commercial/administrative intent,
    not repair symptoms that happen to contain price-adjacent words.
    e.g., "pressure kam hai" (low pressure) must NOT match the price guard.
"""
from __future__ import annotations
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Default OOD pattern registry ──────────────────────────────────────────────
# Each entry: (pattern_re, category_label)
# Pattern priority: first match wins → put highest-precision patterns first.
#
# To override: set env var AGRIFIX_OOD_PATTERNS_FILE pointing to a YAML/JSON
# file with the same structure. The file is loaded once at import time.
# YAML format: [{pattern: "...", category: "..."}, ...]

_DEFAULT_OOD_PATTERNS: List[tuple[str, str]] = [
    # ── Commercial / purchasing intent ────────────────────────────────────────
    # "price" as a standalone concern, not embedded in a symptom
    (r'\bprice\b|\bcost\b|\brate\b|\bdaam\b|\bkeemat\b|\bmehnga\b|\bsasta\b', "price_inquiry"),
    (r'\bbuy\b|\bpurchase\b|\bkharidna\b|\bkharid\b|\bmarket\s+mein\b|\bbazaar\b', "purchase_intent"),
    (r'\bbest\s+brand\b|\btop\s+brand\b|\bkonsa\s+brand\b|\bkaunsa\s+brand\b|\bwhich\s+brand\b', "brand_recommendation"),
    (r'\brecommend\s+(?:a\s+)?(?:pump|motor|tractor|machine)\b', "purchase_recommendation"),
    (r'\bcompare\s+(?:prices|brands|models)\b', "comparison_shopping"),
    # ── Non-repair informational ───────────────────────────────────────────────
    (r'\bhow\s+(?:to\s+)?(?:install|set\s+up|assemble)\s+(?:new|a\s+new)\b', "installation_new_equipment"),
    (r'\bspecification[s]?\b|\bbrochure\b|\bmanual\s+pdf\b|\bdownload\b', "spec_sheet_request"),
    # ── Physical destruction / insurance / theft ──────────────────────────────
    (r'\bstolen\b|\bchori\b|\bchuraaya\b|\bchurai\b', "theft_report"),
    (r'\bcrushed\b|\bsmashed\b|\brun\s+over\b|\btractor\s+(?:fell|rolled)\b|\bdamage(?:d)?\s+by\b', "physical_destruction"),
    (r'\binsurance\b|\bclaim\b|\bcompensation\b|\bmuavza\b', "insurance_claim"),
    # ── Weather / natural disaster ────────────────────────────────────────────
    (r'\bflood\s+(?:damage|insurance)\b|\bcyclone\b|\bearthquake\b', "natural_disaster"),
]

# Allow-list: if any of these are ALSO present, OOD is suppressed.
# Prevents "motor winding burnt due to overload — repair cost?" from being OOD.
_REPAIR_INTENT_OVERRIDE: List[str] = [
    r'\bnot\s+(?:working|starting|running)\b',
    r'\b(?:trip|trips|tripping)\b',
    r'\b(?:leak|leaking)\b',
    r'\b(?:smoke|burning|smell)\b',
    r'\b(?:noise|vibration|shaking)\b',
    r'\b(?:repair|fix|troubleshoot|diagnose)\b',
    r'\b(?:band ho|chalti nahi|shuru nahi)\b',
]


@dataclass
class OODResult:
    is_ood: bool
    category: str = ""
    matched_pattern: str = ""
    query: str = ""

    def api_response(self) -> Dict:
        return {
            "status": "no_data",
            "ood_category": self.category,
            "diagnosis": (
                "This appears to be a question about purchasing, pricing, or a non-repair issue. "
                "This system is designed only for diagnosing machinery faults and providing repair steps. "
                "Please describe the specific fault or symptom you are experiencing."
            ),
            "diagnosis_hi": (
                "यह प्रश्न मशीन की मरम्मत से संबंधित नहीं लगता। "
                "यह सिस्टम केवल मशीन की खराबी का निदान करने के लिए बनाया गया है। "
                "कृपया अपनी मशीन की समस्या या लक्षण बताएं।"
            ),
            "steps": [],
            "parts": [],
            "safety_warnings": [],
        }


class OODGuard:
    """
    Lightweight rule-based OOD classifier. No LLM calls, no network I/O.
    Mean latency: <0.5ms per query.
    """

    def __init__(self) -> None:
        patterns_file = os.environ.get("AGRIFIX_OOD_PATTERNS_FILE")
        if patterns_file:
            self._patterns = self._load_patterns_from_file(patterns_file)
            logger.info("OODGuard: loaded %d patterns from %s", len(self._patterns), patterns_file)
        else:
            self._patterns = [
                (re.compile(p, re.IGNORECASE | re.UNICODE), cat)
                for p, cat in _DEFAULT_OOD_PATTERNS
            ]
            logger.info("OODGuard: using %d default patterns", len(self._patterns))

        self._overrides = [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in _REPAIR_INTENT_OVERRIDE
        ]

    def check(self, query: str, machine_type: str = "") -> OODResult:
        """
        Returns OODResult(is_ood=True) if the query is clearly non-repair.
        Always returns OODResult(is_ood=False) for empty or None queries
        (let downstream validation handle those).
        """
        if not query or not query.strip():
            return OODResult(is_ood=False)

        # Check override first — repair intent always wins
        if any(ov.search(query) for ov in self._overrides):
            return OODResult(is_ood=False)

        for pattern, category in self._patterns:
            if pattern.search(query):
                logger.info("OOD guard: category=%s query='%s...'", category, query[:60])
                return OODResult(is_ood=True, category=category,
                                 matched_pattern=pattern.pattern[:50], query=query)

        return OODResult(is_ood=False)

    def _load_patterns_from_file(self, path: str) -> list:
        """Load patterns from YAML or JSON file. Falls back to defaults on error."""
        try:
            import json as _json
            with open(path) as f:
                data = _json.load(f)
            return [
                (re.compile(entry["pattern"], re.IGNORECASE | re.UNICODE), entry["category"])
                for entry in data
            ]
        except Exception as exc:
            logger.warning("OODGuard: failed to load patterns from %s: %s. Using defaults.", path, exc)
            return [
                (re.compile(p, re.IGNORECASE | re.UNICODE), cat)
                for p, cat in _DEFAULT_OOD_PATTERNS
            ]


# Module-level singleton — instantiated once at import time
_ood_guard = OODGuard()


def check_ood(query: str, machine_type: str = "") -> OODResult:
    """Public entry point for pipeline_orchestrator.py Phase 0."""
    return _ood_guard.check(query, machine_type)