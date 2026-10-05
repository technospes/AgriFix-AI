from __future__ import annotations
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.groq_client import groq_client, TEXT_MODEL, JSON_CONFIG

# FIX D: import single source of truth for the weak-retrieval boundary
from rag import RAG_WEAK_THRESHOLD

logger = logging.getLogger(__name__)

_MAX_CLARIFICATION_ROUNDS = 2

# FIX D: was 0.60 — now uses the unified threshold from rag.py (0.50).
# This ensures clarification and all other confidence gates agree on what
# "weak retrieval" means.
_CLARIFICATION_THRESHOLD = RAG_WEAK_THRESHOLD

# ── Question banks ───────────────────────────────────────────────────────────
_QUESTION_BANK: Dict[str, List[Dict]] = {
    "electrical": [
        {
            "symptom_keywords": ["motor", "electrical", "start", "noise"],
            "question_en": "Are you experiencing any of these electrical symptoms?",
            "question_hi": "Kya aapko inme se koi electrical samasya aa rahi hai?",
            "options_en": ["Motor humming?", "Breaker trips?", "Burn smell?"],
            "options_hi": ["Motor gunguna rahi hai?", "Breaker trip ho raha hai?", "Jalne ki boo aa rahi hai?"],
            "taxonomy_signal": "electrical",
        }
    ],
    "mechanical": [
        {
            "symptom_keywords": ["shaft", "rotate", "stuck", "jammed"],
            "question_en": "Can the shaft rotate manually?",
            "question_hi": "Kya shaft ko haath se ghumaya ja sakta hai?",
            "options_en": ["Yes, it rotates smoothly", "No, it's jammed / hard to turn"],
            "options_hi": ["Haan, aasaani se ghoom raha hai", "Nahi, jaam hai"],
            "taxonomy_signal": "mechanical",
        }
    ],
    "pump": [
        {
            "symptom_keywords": ["water", "flow", "pressure", "discharge"],
            "question_en": "What is the status of the water discharge?",
            "question_hi": "Paani ke bahaav ki sthiti kya hai?",
            "options_en": ["No water discharge", "Air lock / bubbles", "Low pressure"],
            "options_hi": ["Paani nahi nikal raha", "Hawa (Air lock) / bulbule", "Pressure kam hai"],
            "taxonomy_signal": "pump",
        }
    ]
}

_DEFAULT_QUESTIONS = [
    {
        "symptom_keywords": [],
        "question_en": "How long has this problem been happening?",
        "question_hi": "Yeh samasya kitni der se ho rahi hai?",
        "options_en": ["Just started now", "Since yesterday", "For several days"],
        "options_hi": ["Abhi abhi shuru hua", "Kal se", "Kai dino se"],
        "taxonomy_signal": "acute vs chronic fault",
    }
]

_LLM_QUESTION_PROMPT = """\
You are a farm machinery diagnostic assistant.
A farmer reported this problem but the diagnosis is uncertain.
Generate ONE targeted clarifying question to narrow down the cause.

Machine: {machine_type}
Reported symptoms: {symptoms}
Uncertain about: {uncertainty}

Rules:
1. Ask about ONE specific observable symptom (sound, colour, behaviour).
2. Keep it very simple — rural farmer with basic literacy.
3. Provide ONLY 2-3 answer options (button choices, not open-ended).
4. Translate question and options to Hindi as well.
5. Return ONLY JSON — no markdown.

Return EXACTLY:
{{
  "question_en": "<question in plain English>",
  "question_hi": "<same question in simple Hindi>",
  "options_en":  ["<option 1>", "<option 2>"],
  "options_hi":  ["<option 1 Hindi>", "<option 2 Hindi>"]
}}
"""

@dataclass
class ClarificationResult:
    needs_clarification: bool
    question_en:   str               = ""
    question_hi:   str               = ""
    options_en:    List[str]         = field(default_factory=list)
    options_hi:    List[str]         = field(default_factory=list)
    round_number:  int               = 0
    machine_type:  str               = ""
    taxonomy_signal: str             = ""
    source:        str               = ""

    def api_response(self) -> Dict[str, Any]:
        return {
            "status":               "clarification_needed",
            "clarification_round":  self.round_number,
            "question_en":          self.question_en,
            "question_hi":          self.question_hi,
            "options_en":           self.options_en,
            "options_hi":           self.options_hi,
            "machine_type":         self.machine_type,
            "taxonomy_signal":      self.taxonomy_signal,
            "steps":                [],
            "parts":                [],
            "safety_warnings":      [],
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.api_response()


class ClarificationEngine:
    async def get_clarification(
        self,
        machine_type:  str,
        symptoms:      List[str],
        rag_context:   str = "",
        confidence:    float = 0.0,
        round_number:  int = 0,
    ) -> ClarificationResult:
        if round_number >= _MAX_CLARIFICATION_ROUNDS:
            logger.info("Clarification: max rounds reached (%d/%d) — escalating",
                round_number, _MAX_CLARIFICATION_ROUNDS)
            return ClarificationResult(needs_clarification=False,
                machine_type=machine_type, round_number=round_number)

        # FIX D: _CLARIFICATION_THRESHOLD is now RAG_WEAK_THRESHOLD (0.50)
        if confidence >= _CLARIFICATION_THRESHOLD:
            logger.info("Clarification: confidence=%.2f >= threshold=%.2f — not needed",
                confidence, _CLARIFICATION_THRESHOLD)
            return ClarificationResult(needs_clarification=False,
                machine_type=machine_type, round_number=round_number)

        logger.info("Clarification: confidence=%.2f < threshold=%.2f — generating question (round=%d)",
            confidence, _CLARIFICATION_THRESHOLD, round_number)

        bank_result = self._lookup_question_bank(machine_type, symptoms, round_number)
        if bank_result and bank_result.needs_clarification:
            logger.info("Clarification: using bank question (source=%s)", bank_result.source)
            return bank_result

        llm_result = await self._generate_llm_question(machine_type, symptoms, rag_context, round_number)
        if llm_result and llm_result.needs_clarification:
            logger.info("Clarification: using LLM-generated question")
            return llm_result

        logger.warning("Clarification: all question sources failed, using default fallback")
        return self._default_question(machine_type, round_number)

    def _lookup_question_bank(
        self,
        machine_type: str,
        symptoms: List[str],
        round_number: int
    ) -> Optional[ClarificationResult]:
        symptom_text = " ".join(s.lower() for s in symptoms)
        machine_lower = machine_type.lower()
        combined = f"{machine_lower} {symptom_text}"

        _TAXONOMY_SIGNALS: List[Tuple[str, List[str]]] = [
            ("electrical", [
                "motor","electric","winding","capacitor","relay","fuse","mcb",
                "voltage","bijli","current","breaker","alternator","battery",
                "power","avr","genset","generator",
            ]),
            ("pump", [
                "pump","water","pressure","flow","suction","discharge","priming",
                "borewell","submersible","centrifugal","foot valve",
            ]),
            ("mechanical", [
                "shaft","bearing","gear","belt","pulley","impeller","piston",
                "coupling","vibration","jammed","stuck","rotate","tractor",
                "harvester","thresher","rotavator","tine","engine","diesel",
                "crankshaft","valve",
            ]),
        ]

        detected_taxonomy = "mechanical"
        for taxonomy_key, triggers in _TAXONOMY_SIGNALS:
            if any(kw in combined for kw in triggers):
                detected_taxonomy = taxonomy_key
                break

        questions = _QUESTION_BANK.get(detected_taxonomy, [])

        for i, q in enumerate(questions):
            if i < round_number:
                continue
            keywords = q.get("symptom_keywords", [])
            if not keywords or any(kw in symptom_text for kw in keywords):
                return ClarificationResult(
                    needs_clarification=True,
                    question_en=q["question_en"],
                    question_hi=q.get("question_hi", q["question_en"]),
                    options_en=q.get("options_en", [])[:3],
                    options_hi=q.get("options_hi", [])[:3],
                    round_number=round_number,
                    machine_type=machine_type,
                    taxonomy_signal=q.get("taxonomy_signal", ""),
                    source="bank",
                )
        return None

    async def _generate_llm_question(
        self,
        machine_type: str,
        symptoms: List[str],
        rag_context: str,
        round_number: int
    ) -> Optional[ClarificationResult]:
        uncertainty = (
            "multiple possible causes in the manual"
            if rag_context and len(rag_context) > 100
            else "no matching manual entry found yet"
        )
        prompt = _LLM_QUESTION_PROMPT.format(
            machine_type=machine_type.replace("_", " "),
            symptoms="; ".join(symptoms) if symptoms else "unspecified",
            uncertainty=uncertainty
        )
        try:
            raw = await asyncio.to_thread(
                lambda: groq_client.chat.completions.create(
                    model=TEXT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300,
                ).choices[0].message.content
            )
            from utils.json_repair import repair_json
            data = repair_json(raw)
            return ClarificationResult(
                needs_clarification=True,
                question_en=str(data.get("question_en", "")).strip(),
                question_hi=str(data.get("question_hi", "")).strip(),
                options_en=list(data.get("options_en", []))[:3],
                options_hi=list(data.get("options_hi", []))[:3],
                round_number=round_number,
                machine_type=machine_type,
                source="llm",
            )
        except Exception as exc:
            logger.warning("Clarification LLM call failed: %s", exc)
            return None

    def _default_question(self, machine_type: str, round_number: int) -> ClarificationResult:
        q = _DEFAULT_QUESTIONS[0]
        return ClarificationResult(
            needs_clarification=True,
            question_en=q["question_en"],
            question_hi=q["question_hi"],
            options_en=q["options_en"],
            options_hi=q["options_hi"],
            round_number=round_number,
            machine_type=machine_type,
            taxonomy_signal=q.get("taxonomy_signal", ""),
            source="default",
        )


async def get_clarification_if_needed(
    machine_type: str,
    symptoms: List[str],
    confidence: float,
    rag_context: str = "",
    round_number: int = 0,
) -> Optional[Dict[str, Any]]:
    engine = ClarificationEngine()
    result = await engine.get_clarification(
        machine_type=machine_type, symptoms=symptoms,
        rag_context=rag_context, confidence=confidence, round_number=round_number,
    )
    if result.needs_clarification:
        logger.info("Clarification needed: round=%d, source=%s, question='%s'",
            result.round_number, result.source, result.question_en[:50])
        return result.api_response()
    logger.info("Clarification not needed: confidence=%.2f, rounds=%d/%d",
        confidence, round_number, _MAX_CLARIFICATION_ROUNDS)
    return None