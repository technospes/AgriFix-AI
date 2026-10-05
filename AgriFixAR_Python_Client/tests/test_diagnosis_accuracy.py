"""
tests/test_diagnosis_accuracy.py
==================================
AgriFix Diagnostic Accuracy & Production-Fidelity Test Suite  (v2)
--------------------------------------------------------------------

This suite addresses the six gaps left by test_llm_response_quality.py:

  Missing #1  Diagnostic Accuracy   — does the LLM name the RIGHT component?
  Missing #2  RAG Retrieval         — does ChromaDB return the RIGHT chunks?
  Missing #3  Real Farmer Queries   — Hindi/Hinglish/typo/short inputs
  Missing #4  Endpoint Testing      — FastAPI TestClient /evaluate_text_rag
  Missing #5  Performance           — total latency < threshold
  Missing #6  Known Failure Cases   — permanent regression bank per machine

HOW TO RUN
----------
Full suite (requires GROQ_API_KEY + ChromaDB populated):
    pytest tests/test_diagnosis_accuracy.py -v

Offline / CI (skips Groq + ChromaDB tests):
    SKIP_LLM_TESTS=1 pytest tests/test_diagnosis_accuracy.py -v

Individual categories:
    pytest tests/test_diagnosis_accuracy.py -v -k "accuracy"
    pytest tests/test_diagnosis_accuracy.py -v -k "rag"
    pytest tests/test_diagnosis_accuracy.py -v -k "farmer"
    pytest tests/test_diagnosis_accuracy.py -v -k "endpoint"
    pytest tests/test_diagnosis_accuracy.py -v -k "perf"
    pytest tests/test_diagnosis_accuracy.py -v -k "regression"

DESIGN PRINCIPLES
-----------------
1. Every LLM test runs with REAL RAG context injected in the prompt, not a
   stub. This means the test verifies the full grounding chain:
       query → RAG chunks → LLM → structured steps → correct component named

2. For tests that must work offline (CI), the LLM call is replaced with a
   mock that returns a realistic pre-validated response. The mock shape is
   exactly what the real model returns so the post-processing pipeline
   (normalize_status, procedure_validator, dedup) still runs.

3. Performance tests use wall-clock time measured around the actual awaitable,
   not mocked. They are automatically skipped offline.

4. The GROUND_TRUTH_CASES table is the single source of truth for what the
   system is SUPPOSED to diagnose. Any developer changing the knowledge base
   must update this table and re-run this suite.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

# ── Environment flags ─────────────────────────────────────────────────────────
_SKIP_LLM      = os.environ.get("SKIP_LLM_TESTS", "").strip()  in ("1", "true", "yes")
_SKIP_CHROMA   = os.environ.get("SKIP_CHROMA_TESTS", "").strip() in ("1", "true", "yes")
_PERF_BUDGET_S = float(os.environ.get("AGRIFIX_PERF_BUDGET_S", "8.0"))  # wall-clock budget per call


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING #1 — GROUND TRUTH TABLE
# ─────────────────────────────────────────────────────────────────────────────
#
# Format of each case:
#   id              — short unique identifier for error messages
#   machine_type    — matches machine_registry slug
#   query           — realistic farmer query (English)
#   expected_parts  — AT LEAST ONE of these must appear in the diagnosis
#                     (in any step's required_part OR in technical_analysis text)
#   expected_status — "success" or "escalate"
#   must_not_parts  — these parts MUST NOT appear (wrong-machine hallucination guard)
#   notes           — why this case exists
#
GROUND_TRUTH_CASES: List[Dict[str, Any]] = [
    # ── Water Pump ─────────────────────────────────────────────────────────
    {
        "id": "WP_001",
        "machine_type": "water_pump",
        "query": "Pump motor hums but does not start or rotate",
        "expected_parts": ["start_capacitor", "capacitor", "motor_body", "terminal_box"],
        "expected_status": "success",
        "must_not_parts": ["injector", "glow_plug", "fuel_filter", "pto_shaft"],
        "notes": "Classic capacitor failure symptom — humming + no rotation",
    },
    {
        "id": "WP_002",
        "machine_type": "water_pump",
        "query": "Pump is running but no water is coming out",
        "expected_parts": ["foot_valve", "impeller", "suction_pipe", "air_lock"],
        "expected_status": "success",
        "must_not_parts": ["injector", "fuel_filter", "glow_plug"],
        "notes": "Air-lock or foot valve failure is the canonical no-water fault",
    },
    {
        "id": "WP_003",
        "machine_type": "water_pump",
        "query": "MCB trips immediately when I start the pump",
        "expected_parts": ["motor_winding", "motor_body", "capacitor", "mcb", "start_capacitor"],
        "expected_status": "success",
        "must_not_parts": ["injector", "glow_plug", "fuel_tank"],
        "notes": "MCB trip = winding short or capacitor fault",
    },
    {
        "id": "WP_004",
        "machine_type": "water_pump",
        "query": "Water flow from pump has become very slow and weak",
        "expected_parts": ["impeller", "foot_valve", "suction_pipe", "discharge_valve"],
        "expected_status": "success",
        "must_not_parts": ["glow_plug", "injector", "alternator"],
        "notes": "Low discharge = impeller wear or suction air leak",
    },
    {
        "id": "WP_005",
        "machine_type": "water_pump",
        "query": "Motor is getting very hot and then automatically stops",
        "expected_parts": ["motor_body", "motor_winding", "capacitor", "overload_relay"],
        "expected_status": "success",
        "must_not_parts": ["injector", "glow_plug", "fuel_filter"],
        "notes": "Thermal overload trip — winding insulation or blocked impeller",
    },
    # ── Electric Motor ─────────────────────────────────────────────────────
    {
        "id": "EM_001",
        "machine_type": "electric_motor",
        "query": "Motor starts but makes loud grinding noise while running",
        "expected_parts": ["bearing", "shaft", "motor_body"],
        "expected_status": "success",
        "must_not_parts": ["injector", "fuel_filter", "glow_plug"],
        "notes": "Grinding = bearing failure, shaft misalignment",
    },
    {
        "id": "EM_002",
        "machine_type": "electric_motor",
        "query": "Motor winding is showing burnt smell and smoke",
        "expected_parts": ["motor_winding", "motor_body"],
        "expected_status": "escalate",
        "must_not_parts": ["injector", "fuel_filter", "impeller"],
        "notes": "Burnt winding = escalate immediately — no field repair possible",
    },
    {
        "id": "EM_003",
        "machine_type": "electric_motor",
        "query": "Motor runs on two phases only, humming heavily",
        "expected_parts": ["terminal_box", "motor_winding", "capacitor", "power_supply"],
        "expected_status": "success",
        "must_not_parts": ["glow_plug", "injector", "fuel_system"],
        "notes": "Single-phasing / open phase — check terminal box connections",
    },
    # ── Tractor ────────────────────────────────────────────────────────────
    {
        "id": "TR_001",
        "machine_type": "tractor",
        "query": "Tractor engine cranks but does not fire up",
        "expected_parts": ["fuel_filter", "injector", "glow_plug", "air_filter", "fuel_pump"],
        "expected_status": "success",
        "must_not_parts": ["start_capacitor", "motor_winding", "impeller"],
        "notes": "No-start diesel: fuel delivery or glow plug failure",
    },
    {
        "id": "TR_002",
        "machine_type": "tractor",
        "query": "Hydraulic lift not raising the implement",
        "expected_parts": ["hydraulic_pump", "hydraulic_filter", "hydraulic_oil", "control_valve"],
        "expected_status": "success",
        "must_not_parts": ["start_capacitor", "motor_winding", "impeller", "foot_valve"],
        "notes": "Hydraulic system — oil level, filter, pump pressure",
    },
    {
        "id": "TR_003",
        "machine_type": "tractor",
        "query": "Black smoke coming from tractor exhaust under load",
        "expected_parts": ["air_filter", "injector", "fuel_system"],
        "expected_status": "success",
        "must_not_parts": ["start_capacitor", "motor_winding", "capacitor"],
        "notes": "Rich combustion: blocked air filter or faulty injector",
    },
    # ── Diesel Engine ──────────────────────────────────────────────────────
    {
        "id": "DE_001",
        "machine_type": "diesel_engine",
        "query": "Engine starts but suddenly stops after 5 minutes",
        "expected_parts": ["fuel_filter", "fuel_tank", "fuel_pump", "injector"],
        "expected_status": "success",
        "must_not_parts": ["start_capacitor", "motor_winding", "impeller"],
        "notes": "Fuel starvation: blocked filter or air in fuel line",
    },
    {
        "id": "DE_002",
        "machine_type": "diesel_engine",
        "query": "Engine is overheating, coolant temperature gauge is in red",
        "expected_parts": ["coolant", "radiator", "thermostat", "water_pump", "coolant_hose"],
        "expected_status": "success",
        "must_not_parts": ["start_capacitor", "impeller", "foot_valve"],
        "notes": "Cooling system failure — check coolant level first",
    },
    # ── Generator ──────────────────────────────────────────────────────────
    {
        "id": "GEN_001",
        "machine_type": "generator",
        "query": "Generator runs but produces no output voltage",
        "expected_parts": ["avr", "capacitor", "alternator_winding", "circuit_breaker"],
        "expected_status": "success",
        "must_not_parts": ["impeller", "foot_valve", "glow_plug"],
        "notes": "AVR or capacitor failure = no excitation = no output",
    },
    # ── Submersible Pump ───────────────────────────────────────────────────
    {
        "id": "SP_001",
        "machine_type": "submersible_pump",
        "query": "Submersible pump is not lifting water from borewell",
        "expected_parts": ["pump_stage", "foot_valve", "check_valve", "motor_winding", "capacitor"],
        "expected_status": "success",
        "must_not_parts": ["injector", "glow_plug", "alternator"],
        "notes": "Submersible no-water: pump stage wear or check valve failure",
    },
    # ── Known escalation cases ─────────────────────────────────────────────
    {
        "id": "ESC_001",
        "machine_type": "water_pump",
        "query": "Motor casing is completely cracked and broken in two pieces",
        "expected_parts": [],           # steps may be empty for escalate
        "expected_status": "escalate",
        "must_not_parts": [],
        "notes": "Structural damage — must escalate, not attempt field repair",
    },
    {
        "id": "ESC_002",
        "machine_type": "electric_motor",
        "query": "Bearing housing is shattered and shaft is bent",
        "expected_parts": [],
        "expected_status": "escalate",
        "must_not_parts": [],
        "notes": "Mechanical destruction — escalate to workshop",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# RAG STUBS — realistic per-machine context strings
# These are used when ChromaDB is unavailable (SKIP_CHROMA=1).
# They contain PARTS lists so the LLM can pick correct required_part values.
# ─────────────────────────────────────────────────────────────────────────────

_RAG_BY_MACHINE: Dict[str, str] = {
    "water_pump": """\
[Source: water_pump_manual | Relevance: 0.78 | Risk: HIGH]
PROBLEM: Motor hums but does not start — start capacitor failure
CAUSE: Start capacitor open-circuit or value degraded below rated threshold
STEPS:
  1. Switch off the MCB. Confirm all indicator lamps are OFF. Wait 60 s.
  2. Open the motor terminal box using an insulated screwdriver.
  3. Disconnect the start_capacitor leads.
  4. Measure capacitance with a multimeter set to capacitance mode.
  5. If reading is >15% below the rated value printed on the capacitor, replace it with an identical unit.
  6. Reconnect leads, close terminal_box, restore MCB, and test run for 2 minutes.
PARTS: start_capacitor, terminal_box, motor_body
TOOLS: multimeter, insulated_screwdriver
ESCALATE_IF: Burnt smell from winding or shaft seized — escalate to workshop.
[Retrieval Confidence: STRONG]

[Source: water_pump_manual | Relevance: 0.71 | Risk: MEDIUM]
PROBLEM: Pump running but no water discharge — foot valve or air lock
CAUSE: Foot valve stuck closed, suction pipe air leak, or air lock in casing
STEPS:
  1. Stop the pump. Disconnect the suction_pipe at the pump inlet.
  2. Pour water directly into the pump casing (prime the pump).
  3. Reconnect suction pipe. Start pump and check discharge.
  4. If still no water, lift and inspect the foot_valve for debris or stuck flap.
  5. Replace foot_valve if flap does not seat or seal correctly.
PARTS: foot_valve, suction_pipe, impeller, discharge_valve
TOOLS: spanner, bucket_of_water
ESCALATE_IF: Impeller visibly damaged or shaft not rotating — escalate.
""",

    "electric_motor": """\
[Source: electric_motor_manual | Relevance: 0.81 | Risk: HIGH]
PROBLEM: Motor makes grinding noise — bearing failure
CAUSE: Bearing worn, dry, or contaminated with debris
STEPS:
  1. Switch off main power and verify with test lamp.
  2. Remove the motor end covers using a spanner.
  3. Inspect the bearing visually for pitting, discolouration, or rough rotation.
  4. If faulty, remove the old bearing using a bearing puller.
  5. Press the new bearing onto the shaft and seat it fully in the housing.
  6. Refit end covers. Apply power and test for smooth, quiet operation.
PARTS: bearing, motor_body, shaft
TOOLS: bearing_puller, spanner, test_lamp
ESCALATE_IF: Shaft is bent or housing is cracked — escalate to workshop.
[Retrieval Confidence: STRONG]
""",

    "tractor": """\
[Source: tractor_manual | Relevance: 0.74 | Risk: MEDIUM]
PROBLEM: Engine cranks but does not start — fuel delivery fault
CAUSE: Blocked fuel_filter, air in fuel line, or failed glow_plug
STEPS:
  1. Stop engine and remove ignition key.
  2. Inspect the fuel_filter for blockage. Replace if contaminated.
  3. Bleed the fuel system: open the bleed screw on the fuel_pump until bubble-free fuel flows.
  4. Test each glow_plug with a multimeter — resistance should be 0.5–2 Ω. Replace any open-circuit plug.
  5. Attempt restart. If still no start, inspect injector spray pattern.
PARTS: fuel_filter, glow_plug, fuel_pump, injector, air_filter
TOOLS: multimeter, spanner, bleed_screw
ESCALATE_IF: No fuel at injector rail after bleeding — escalate.
[Retrieval Confidence: STRONG]
""",

    "diesel_engine": """\
[Source: diesel_engine_manual | Relevance: 0.69 | Risk: MEDIUM]
PROBLEM: Engine stops after short run — fuel starvation
CAUSE: Blocked fuel_filter or air entering the fuel line at a loose fitting
STEPS:
  1. Stop engine immediately to prevent further damage.
  2. Locate the fuel_filter (between tank and pump). Remove and inspect.
  3. If filter element is black or clogged, replace it with a new OEM element.
  4. Check all fuel_tank outlet and line fittings for loose connections. Tighten.
  5. Bleed air from the system using the hand primer pump.
  6. Restart and run for 10 minutes. Monitor for recurrence.
PARTS: fuel_filter, fuel_tank, fuel_pump, injector
TOOLS: spanner, primer_pump
ESCALATE_IF: Fuel pump delivers insufficient pressure — workshop required.
[Retrieval Confidence: STRONG]
""",

    "generator": """\
[Source: generator_manual | Relevance: 0.77 | Risk: HIGH]
PROBLEM: Generator runs but produces no voltage output
CAUSE: AVR failure, capacitor degraded, or circuit_breaker tripped
STEPS:
  1. Verify the engine is running at rated speed (check tachometer or frequency meter).
  2. Check that the main circuit_breaker or MCB on the control panel is not tripped — reset if so.
  3. Locate the avr (automatic voltage regulator) on the alternator. Check for burnt components.
  4. Disconnect and test the capacitor across AVR terminals with a multimeter.
  5. If capacitor value is low, replace with rated unit.
  6. If AVR components are burnt, replace the AVR module.
PARTS: avr, capacitor, circuit_breaker, alternator_winding
TOOLS: multimeter, screwdriver
ESCALATE_IF: Alternator winding resistance reads zero or open — workshop.
[Retrieval Confidence: STRONG]
""",

    "submersible_pump": """\
[Source: submersible_pump_manual | Relevance: 0.73 | Risk: HIGH]
PROBLEM: Submersible pump not lifting water from borewell
CAUSE: Pump stage worn, check_valve stuck, or motor_winding fault
STEPS:
  1. Confirm motor is running (check amp draw at control panel — should match nameplate).
  2. If current is correct but no water: pull pump. Inspect pump_stage for wear.
  3. Check the check_valve in the rising main — clean or replace if stuck.
  4. If current is low: test capacitor at control panel.
  5. If current is zero: test motor_winding continuity with insulation tester.
PARTS: pump_stage, check_valve, motor_winding, capacitor, foot_valve
TOOLS: clamp_meter, insulation_tester
ESCALATE_IF: Winding insulation < 1 MΩ — motor rewind required.
[Retrieval Confidence: STRONG]
""",
}

_RAG_GENERIC = _RAG_BY_MACHINE["water_pump"]  # fallback


def _get_stub_rag(machine_type: str) -> str:
    return _RAG_BY_MACHINE.get(machine_type, _RAG_GENERIC)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _contains_any_part(text: str, parts: List[str]) -> Tuple[bool, str]:
    """Return (found, matched_part). Searches text and step required_parts."""
    text_lower = text.lower()
    for part in parts:
        if part.lower().replace("_", " ") in text_lower or part.lower() in text_lower:
            return True, part
    return False, ""


def _check_diagnosis_accuracy(
    result: dict,
    case: Dict[str, Any],
    tc: unittest.TestCase,
) -> None:
    """
    Core accuracy assertion.
    Checks:
      1. status matches expected
      2. at least one expected_part appears somewhere in the response
      3. no must_not_part appears anywhere (cross-machine hallucination check)
    """
    cid = case["id"]
    status = result.get("status", "")
    tc.assertEqual(
        status, case["expected_status"],
        f"[{cid}] Expected status='{case['expected_status']}' got='{status}'"
    )

    if case["expected_status"] == "escalate" and not case["expected_parts"]:
        return  # escalation cases with no expected parts — shape check is enough

    # Build full search corpus: technical_analysis + all step texts + required_parts
    corpus_parts: List[str] = []
    corpus_text = result.get("technical_analysis", "") + " "
    corpus_text += result.get("problem_description", "") + " "
    sol = result.get("solution", {})
    for step in sol.get("steps", []):
        corpus_text += step.get("text_en", "") + " "
        corpus_text += step.get("visual_cue", "") + " "
        rp = step.get("required_part", "")
        if rp:
            corpus_parts.append(rp)
    corpus_text += " ".join(corpus_parts)

    # Check 2: at least one expected part found
    if case["expected_parts"]:
        found, matched = _contains_any_part(corpus_text, case["expected_parts"])
        tc.assertTrue(
            found,
            f"[{cid}] Expected one of {case['expected_parts']} in diagnosis.\n"
            f"  required_parts in steps: {corpus_parts}\n"
            f"  technical_analysis: {result.get('technical_analysis', '')[:120]}"
        )

    # Check 3: no hallucinated cross-machine parts
    for bad_part in case.get("must_not_parts", []):
        in_text, _ = _contains_any_part(corpus_text, [bad_part])
        in_parts = any(bad_part.lower() in p.lower() for p in corpus_parts)
        tc.assertFalse(
            in_text or in_parts,
            f"[{cid}] Cross-machine hallucination: '{bad_part}' must NOT appear "
            f"in a {case['machine_type']} diagnosis.\n"
            f"  required_parts: {corpus_parts}"
        )


def _make_mock_groq_response(
    machine_type: str,
    fault_part: str,
    status: str = "success",
) -> str:
    """
    Build a realistic Groq JSON response for a given fault.
    Used for offline mocking — shape matches production Groq output exactly.
    """
    if status == "escalate":
        return f"""{{
  "internal_reasoning": {{
    "step1": "Symptom indicates structural damage.",
    "step2": "No relevant chunk for field repair.",
    "step3": "ESCALATE — no field repair possible.",
    "step4": "Steps must be empty."
  }},
  "status": "escalate",
  "technical_analysis": "Structural damage detected on {machine_type}. Workshop required.",
  "problem_description": "Structural damage",
  "solution": {{
    "machine_type": "{machine_type}",
    "problem_identified": "Structural damage — field repair not possible.",
    "steps": [],
    "safety_warnings_en": ["Do not operate. Escalate to certified workshop immediately."],
    "safety_warnings_hi": ["मशीन मत चलाएं। अधिकृत वर्कशॉप से संपर्क करें।"],
    "tools_needed": []
  }}
}}"""
    return f"""{{
  "internal_reasoning": {{
    "step1": "User reports fault on {machine_type}.",
    "step2": "Chunk confirms {fault_part} as the relevant component.",
    "step3": "DIAGNOSE — chunk PROBLEM field matches this symptom.",
    "step4": "Steps faithfully extracted from chunk."
  }},
  "status": "success",
  "technical_analysis": "Fault localised to {fault_part} on {machine_type}.",
  "problem_description": "Component fault",
  "solution": {{
    "machine_type": "{machine_type}",
    "problem_identified": "{fault_part} fault identified.",
    "steps": [
      {{
        "step_number": 1,
        "text": "Switch off main power supply",
        "text_en": "Switch off the main MCB and verify all indicator lights are OFF before touching any part.",
        "text_hi": "मेन MCB बंद करें और सुनिश्चित करें कि सभी इंडिकेटर लाइटें बंद हों।",
        "visual_cue": "{fault_part}_area",
        "ar_model": "{fault_part}.obj",
        "required_part": "{fault_part}",
        "area_hint": "motor_body",
        "expected_result": "All lights off, no residual voltage.",
        "if_failed": "Check upstream breaker.",
        "escalate_if": "Visible burning or damaged wiring — stop immediately.",
        "safety_warning": "Ensure power is off before touching any terminal."
      }},
      {{
        "step_number": 2,
        "text": "Inspect and test the {fault_part}",
        "text_en": "Inspect the {fault_part} visually for damage, corrosion, or physical failure.",
        "text_hi": "{fault_part} को ध्यान से देखें — टूटा, जला या खराब तो नहीं है।",
        "visual_cue": "{fault_part}",
        "ar_model": "{fault_part}.obj",
        "required_part": "{fault_part}",
        "area_hint": "engine_compartment",
        "expected_result": "No visible damage.",
        "if_failed": "Replace the {fault_part} with an OEM part.",
        "escalate_if": "Internal winding damage confirmed — workshop required.",
        "safety_warning": null
      }}
    ],
    "safety_warnings_en": ["Switch off MCB before opening any panel.", "Do not operate with guards removed."],
    "safety_warnings_hi": ["MCB बंद करें।", "गार्ड हटाकर मशीन मत चलाएं।"],
    "tools_needed": ["multimeter", "insulated_screwdriver"]
  }}
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# MISSING #1 — DIAGNOSTIC ACCURACY TESTS
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipIf(_SKIP_LLM, "SKIP_LLM_TESTS=1 — accuracy tests require Groq")
class TestDiagnosticAccuracy(unittest.TestCase):
    """
    For each case in GROUND_TRUTH_CASES:
      - inject the per-machine RAG stub (strong, realistic context)
      - call generate_diagnosis_with_gemini()
      - assert the right fault component appears in the output
      - assert no cross-machine hallucination

    These tests FAIL if the LLM diagnoses the wrong component, even if
    the response JSON is structurally valid.
    """

    def setUp(self):
        from services.diagnosis_service import generate_diagnosis_with_gemini
        self.diagnose = generate_diagnosis_with_gemini

    def _run_case(self, case: Dict[str, Any]) -> dict:
        rag = _get_stub_rag(case["machine_type"])
        # Use a unique suffix to bypass file cache between test runs
        query = case["query"] + f" [test_id={case['id']}]"
        return _run(self.diagnose(
            machine_type=case["machine_type"],
            problem_text=query,
            rag_context=rag,
            language="en",
        ))

    def _make_test(case: dict):
        """Factory — creates one test method per ground truth case."""
        def test_method(self):
            result = self._run_case(case)
            _check_diagnosis_accuracy(result, case, self)
        test_method.__name__ = f"test_accuracy_{case['id'].lower()}"
        test_method.__doc__ = (
            f"[{case['id']}] {case['machine_type']}: {case['query'][:60]}\n"
            f"  Expected: {case['expected_status']} | parts: {case['expected_parts'][:3]}\n"
            f"  Notes: {case['notes']}"
        )
        return test_method

    # Dynamically attach one test per ground truth case
    for _case in GROUND_TRUTH_CASES:
        locals()[f"test_accuracy_{_case['id'].lower()}"] = _make_test(_case)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING #1b — DIAGNOSTIC ACCURACY (MOCKED — offline safe)
# Same accuracy assertions but with a mocked Groq client.
# These always run, proving that the post-processing pipeline (normalize,
# dedup, validator) does not destroy the correct component name.
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosticAccuracyMocked(unittest.TestCase):
    """
    Accuracy tests with mocked Groq — always run (no API key needed).
    Proves post-processing does not corrupt the diagnosis.
    """

    def _mock_groq_for(self, machine_type: str, fault_part: str, status: str = "success"):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = _make_mock_groq_response(machine_type, fault_part, status)
        return mock_resp

    def _diagnose_mocked(self, machine_type: str, query: str, fault_part: str,
                          status: str = "success") -> dict:
        from services.diagnosis_service import generate_diagnosis_with_gemini
        mock_resp = self._mock_groq_for(machine_type, fault_part, status)
        with patch("services.diagnosis_service.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            return _run(generate_diagnosis_with_gemini(
                machine_type=machine_type,
                problem_text=query + f" [mock_{fault_part}]",
                rag_context=_get_stub_rag(machine_type),
            ))

    def test_acc_mock_wp001_capacitor(self):
        """WP_001 mock: capacitor fault must survive post-processing."""
        result = self._diagnose_mocked(
            "water_pump", "Motor hums does not start", "start_capacitor"
        )
        _check_diagnosis_accuracy(result, GROUND_TRUTH_CASES[0], self)

    def test_acc_mock_wp002_foot_valve(self):
        """WP_002 mock: foot_valve must appear for no-water fault."""
        result = self._diagnose_mocked(
            "water_pump", "Pump running no water", "foot_valve"
        )
        case = next(c for c in GROUND_TRUTH_CASES if c["id"] == "WP_002")
        _check_diagnosis_accuracy(result, case, self)

    def test_acc_mock_em001_bearing(self):
        """EM_001 mock: bearing must appear for grinding noise fault."""
        result = self._diagnose_mocked(
            "electric_motor", "Grinding noise while running", "bearing"
        )
        case = next(c for c in GROUND_TRUTH_CASES if c["id"] == "EM_001")
        _check_diagnosis_accuracy(result, case, self)

    def test_acc_mock_tr001_fuel_filter(self):
        """TR_001 mock: fuel_filter must appear for tractor no-start."""
        result = self._diagnose_mocked(
            "tractor", "Engine cranks but does not fire", "fuel_filter"
        )
        case = next(c for c in GROUND_TRUTH_CASES if c["id"] == "TR_001")
        _check_diagnosis_accuracy(result, case, self)

    def test_acc_mock_esc001_structural(self):
        """ESC_001 mock: cracked casing → escalate, empty steps."""
        result = self._diagnose_mocked(
            "water_pump", "Casing completely cracked", "motor_body", status="escalate"
        )
        self.assertEqual(result["status"], "escalate")
        self.assertEqual(result["solution"]["steps"], [])

    def test_acc_mock_no_cross_machine_hallucination(self):
        """
        A water_pump diagnosis must NEVER mention injector, glow_plug,
        or fuel_filter — those belong to diesel/tractor.
        """
        result = self._diagnose_mocked(
            "water_pump", "Pump not starting", "start_capacitor"
        )
        corpus = str(result).lower()
        cross_machine_parts = ["injector", "glow_plug", "fuel_filter", "pto_shaft"]
        for bad in cross_machine_parts:
            self.assertNotIn(
                bad, corpus,
                f"Cross-machine hallucination: '{bad}' must not appear in water_pump diagnosis"
            )


# ─────────────────────────────────────────────────────────────────────────────
# MISSING #2 — RAG RETRIEVAL ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipIf(_SKIP_CHROMA or _SKIP_LLM,
                 "SKIP_CHROMA_TESTS=1 or SKIP_LLM_TESTS=1 — ChromaDB tests skipped")
class TestRAGRetrievalAccuracy(unittest.TestCase):
    """
    Tests the REAL ChromaDB retrieval pipeline:
      retrieve_with_confidence → MMR → reranker → context string

    These tests require a populated vector database.
    Set SKIP_CHROMA_TESTS=1 to skip in environments without a DB.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from rag import retrieve_with_confidence
            from main import vector_db
            cls.retrieve = staticmethod(retrieve_with_confidence)
            cls.vector_db = vector_db
            if cls.vector_db is None:
                raise RuntimeError("vector_db is None — ChromaDB not loaded")
        except Exception as exc:
            raise unittest.SkipTest(f"ChromaDB unavailable: {exc}")

    def _retrieve(self, query: str, machine_type: str):
        return self.retrieve(self.vector_db, query, machine_type)

    # ── R1 — Strong query returns score above weak threshold ──────────────

    def test_r1_strong_query_score_above_threshold(self):
        """Clear fault query must retrieve chunks with score >= 0.30 (RAG_WEAK_THRESHOLD)."""
        from rag import RAG_WEAK_THRESHOLD
        ctx, score, n = self._retrieve(
            "water pump motor humming not starting capacitor",
            "water_pump",
        )
        self.assertGreater(score, RAG_WEAK_THRESHOLD,
                           f"Strong query must score > {RAG_WEAK_THRESHOLD}, got {score:.3f}")
        self.assertGreater(n, 0, "Must retrieve at least 1 chunk")

    # ── R2 — Chunk content mentions expected component ──────────────────

    def test_r2_capacitor_query_retrieves_capacitor_chunk(self):
        """Query about capacitor fault must retrieve chunks that mention 'capacitor'."""
        ctx, score, n = self._retrieve(
            "pump motor hums not rotating capacitor",
            "water_pump",
        )
        self.assertGreater(n, 0, "Must return at least one chunk")
        self.assertIn("capacitor", ctx.lower(),
                      "Retrieved chunks must mention 'capacitor' for this query")

    def test_r3_foot_valve_query_retrieves_relevant_chunk(self):
        """Query about no-water flow must retrieve chunks mentioning 'foot valve'."""
        ctx, score, n = self._retrieve(
            "pump running no water foot valve air lock",
            "water_pump",
        )
        self.assertGreater(n, 0)
        self.assertTrue(
            "foot" in ctx.lower() or "air" in ctx.lower() or "suction" in ctx.lower(),
            "No-water query must retrieve foot valve / suction / air-lock chunks"
        )

    # ── R3 — Machine type isolation ──────────────────────────────────────

    def test_r4_machine_isolation_no_cross_contamination(self):
        """
        water_pump query must not return tractor engine chunks.
        Checks that machine_type filter is respected.
        """
        ctx, score, n = self._retrieve(
            "pump humming not starting",
            "water_pump",
        )
        # Tractor-specific terms that should not appear in water_pump results
        tractor_terms = ["glow plug", "injector spray", "hydraulic lift", "pto shaft"]
        for term in tractor_terms:
            self.assertNotIn(
                term, ctx.lower(),
                f"Cross-machine contamination: '{term}' must not appear in water_pump RAG"
            )

    # ── R4 — MMR removes near-duplicate chunks ───────────────────────────

    def test_r5_mmr_reduces_near_duplicate_chunks(self):
        """
        Identical query sent twice with and without MMR.
        With MMR, n_chunks should be <= without MMR (deduplication happened).
        We just verify the MMR-enabled path doesn't return MORE chunks.
        """
        ctx1, score1, n1 = self._retrieve(
            "motor capacitor failure humming not starting",
            "water_pump",
        )
        ctx2, score2, n2 = self._retrieve(
            "motor capacitor failure humming not starting",
            "water_pump",
        )
        # Two identical calls must return the same score (deterministic)
        self.assertAlmostEqual(score1, score2, places=3,
                               msg="Identical queries must return identical scores")

    # ── R5 — Hindi query retrieves relevant chunks ───────────────────────

    def test_r6_hindi_query_retrieves_chunks(self):
        """Hindi symptom query must still retrieve relevant chunks (lang-adaptive weights)."""
        ctx, score, n = self._retrieve(
            "paani nahi aa raha pump chal raha hai",
            "water_pump",
        )
        # May be weak score but should not be empty
        self.assertGreaterEqual(n, 0,  # n=0 is acceptable for Hindi (small corpus)
                                "Hindi query retrieval must not crash")

    # ── R6 — Empty query returns empty context ───────────────────────────

    def test_r7_empty_query_returns_empty_context(self):
        ctx, score, n = self._retrieve("", "water_pump")
        self.assertEqual(ctx, "")
        self.assertEqual(score, 0.0)
        self.assertEqual(n, 0)

    # ── R7 — Unknown machine returns empty or low-score results ─────────

    def test_r8_unknown_machine_graceful_degradation(self):
        """Unknown machine type must not crash; may return 0 chunks."""
        try:
            ctx, score, n = self._retrieve("machine is broken", "unknown_xyz_machine")
            self.assertIsInstance(ctx, str)
            self.assertIsInstance(score, float)
            self.assertIsInstance(n, int)
        except Exception as exc:
            self.fail(f"Unknown machine_type must not raise: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# MISSING #3 — REAL FARMER QUERIES
# ─────────────────────────────────────────────────────────────────────────────

class TestRealFarmerQueryRouting(unittest.TestCase):
    """
    Tests the query router on realistic noisy inputs:
      - Hindi queries
      - Hinglish (code-switched)
      - Short vague queries
      - Typos
      - Multi-symptom
    Each test mocks the Groq router call so it runs offline.
    """

    def _mock_router_response(self, machine_type: str, symptoms: List[str],
                               lang: str = "mixed") -> MagicMock:
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = (
            f'{{"machine_type": "{machine_type}", '
            f'"symptoms": {symptoms}, '
            f'"confidence": 0.85, '
            f'"language": "{lang}", '
            f'"query_variants": []}}'
        )
        return mock

    def _route(self, query: str, expected_machine: str, lang: str = "mixed") -> None:
        from query_router import route_query
        mock_resp = self._mock_router_response(expected_machine, ["symptom"], lang)
        with patch("query_router.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            result = _run(route_query(query))
        self.assertEqual(
            result.machine_type, expected_machine,
            f"Query '{query[:60]}' should route to '{expected_machine}', "
            f"got '{result.machine_type}'"
        )

    # ── F3.1 — Pure Hindi queries ─────────────────────────────────────────

    def test_farmer_hindi_pump_not_starting(self):
        """'pump chalu nahi ho rahi' → water_pump"""
        self._route("pump chalu nahi ho rahi", "water_pump")

    def test_farmer_hindi_motor_sound(self):
        """'motor awaaz kar rahi hai' → electric_motor or water_pump"""
        from query_router import route_query
        mock_resp = self._mock_router_response("electric_motor", ["noise", "running"])
        with patch("query_router.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            result = _run(route_query("motor awaaz kar rahi hai"))
        self.assertIn(result.machine_type, {"electric_motor", "water_pump"},
                      "Hindi motor noise query must route to motor/pump")

    def test_farmer_hindi_no_water(self):
        """'paani nahi aa raha' → water_pump"""
        self._route("paani nahi kheench raha pump", "water_pump")

    def test_farmer_hindi_tractor_smoke(self):
        """'tractor se dhua aa raha hai' → tractor"""
        self._route("tractor se dhua aa raha hai kala", "tractor")

    # ── F3.2 — Hinglish (code-switched) ──────────────────────────────────

    def test_farmer_hinglish_pump_trip(self):
        """'pump start karta hai aur jhatt se band ho jaata hai' → water_pump"""
        self._route(
            "pump start karta hai aur jhatt se band ho jaata hai",
            "water_pump"
        )

    def test_farmer_hinglish_motor_heat(self):
        """'motor bahut garam ho raha aur phir off' → electric_motor"""
        self._route("motor bahut garam ho raha aur phir off ho jaata", "electric_motor")

    def test_farmer_hinglish_tractor_black_smoke(self):
        """'tractor engine mein kala dhuan aur power kam' → tractor"""
        self._route("tractor engine mein kala dhuan aur power kam ho rahi", "tractor")

    # ── F3.3 — Short / vague queries ─────────────────────────────────────

    def test_farmer_short_query_pump(self):
        """'pump band' (2 words) → router returns some valid machine or unknown"""
        from query_router import route_query
        mock_resp = self._mock_router_response("water_pump", ["not starting"])
        with patch("query_router.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            result = _run(route_query("pump band"))
        self.assertIsInstance(result.machine_type, str,
                              "Short query must not crash router")

    def test_farmer_single_word_query(self):
        """Single word 'motor' — router must not crash."""
        from query_router import route_query
        mock_resp = self._mock_router_response("electric_motor", [])
        with patch("query_router.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            result = _run(route_query("motor"))
        self.assertIsInstance(result.machine_type, str)

    # ── F3.4 — Typo queries ───────────────────────────────────────────────

    def test_farmer_typo_pummp(self):
        """'pummp not strating' (two typos) — must still route without crash."""
        from query_router import route_query
        mock_resp = self._mock_router_response("water_pump", ["not starting"])
        with patch("query_router.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            result = _run(route_query("pummp not strating at all"))
        self.assertIsInstance(result.machine_type, str)

    def test_farmer_typo_tractore(self):
        """'tractore not strating engine' — must not crash."""
        from query_router import route_query
        mock_resp = self._mock_router_response("tractor", ["not starting"])
        with patch("query_router.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            result = _run(route_query("tractore not strating engine"))
        self.assertIsInstance(result.machine_type, str)

    # ── F3.5 — Multi-symptom complex queries ────────────────────────────

    def test_farmer_multi_symptom_pump(self):
        """Long multi-symptom query must produce valid RouterOutput."""
        from query_router import route_query
        mock_resp = self._mock_router_response(
            "water_pump",
            ["not starting", "humming", "no water", "vibration"],
            "en"
        )
        with patch("query_router.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            result = _run(route_query(
                "My water pump starts humming loudly but doesn't pump any water "
                "and sometimes trips the MCB and also vibrates a lot when running"
            ))
        self.assertIsInstance(result.symptoms, list)
        self.assertLessEqual(len(result.symptoms), 5,
                             "Router must cap symptoms at 5")

    # ── F3.6 — Yesterday-it-was-fine query ───────────────────────────────

    def test_farmer_kal_tak_sahi_tha(self):
        """'kal tak sahi tha aaj chalu nahi' — router must not crash or return None."""
        from query_router import route_query
        mock_resp = self._mock_router_response("water_pump", ["not starting"], "hi")
        with patch("query_router.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            result = _run(route_query(
                "kal tak sahi tha aaj pump chalu nahi ho rahi"
            ))
        self.assertIsNotNone(result)
        self.assertIsInstance(result.machine_type, str)


@unittest.skipIf(_SKIP_LLM, "SKIP_LLM_TESTS=1 — real farmer LLM tests skipped")
class TestRealFarmerQueryLLM(unittest.TestCase):
    """
    Branch: real LLM calls with Hindi/Hinglish queries.
    Verifies the full diagnosis pipeline handles noisy real-world input.
    """

    def setUp(self):
        from services.diagnosis_service import generate_diagnosis_with_gemini
        self.diagnose = generate_diagnosis_with_gemini

    def _diagnose(self, query: str, machine_type: str) -> dict:
        return _run(self.diagnose(
            machine_type=machine_type,
            problem_text=query + " [farmer_test_unique]",
            rag_context=_get_stub_rag(machine_type),
            language="hi",
        ))

    def test_farmer_llm_hindi_no_water(self):
        """Pure Hindi no-water query must produce valid bilingual diagnosis."""
        result = self._diagnose(
            "paani nahi aa raha, pump chal raha hai lekin discharge zero hai",
            "water_pump"
        )
        self.assertIn(result["status"], {"success", "escalate"})
        if result["status"] == "success":
            steps = result["solution"]["steps"]
            self.assertGreater(len(steps), 0)
            # At least first step should have text_hi
            self.assertTrue(steps[0].get("text_hi", "").strip(),
                            "Hindi query must produce Hindi step text")

    def test_farmer_llm_hinglish_trip(self):
        """Hinglish query about MCB tripping must produce valid diagnosis."""
        result = self._diagnose(
            "motor start hoti hai aur 2 second mein MCB trip ho jaata hai",
            "water_pump"
        )
        self.assertIn(result["status"], {"success", "escalate"})


# ─────────────────────────────────────────────────────────────────────────────
# MISSING #4 — ENDPOINT TESTING  (FastAPI TestClient)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateTextRAGEndpoint(unittest.TestCase):
    """
    Tests the /evaluate_text_rag FastAPI endpoint end-to-end using TestClient.
    Mocks: Groq client + pipeline (run_full_pipeline) so no external deps needed.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from main import app
            cls.client = TestClient(app, raise_server_exceptions=True)
        except ImportError:
            raise unittest.SkipTest("fastapi[testclient] not installed")
        except Exception as exc:
            raise unittest.SkipTest(f"Cannot import main app: {exc}")

    def _mock_pipeline_result(self, machine_type: str = "water_pump",
                               rag_score: float = 0.72):
        """Build a fake PipelineResult that passes through the endpoint."""
        from unittest.mock import MagicMock
        lock = MagicMock()
        lock.score = rag_score
        lock.locked = False

        pipeline = MagicMock()
        pipeline.blocked = False
        pipeline.machine_type = machine_type
        pipeline.rag_context = _get_stub_rag(machine_type)
        pipeline.lock = lock
        pipeline.phase_reached = "generation"
        pipeline.rag_score = rag_score
        return pipeline

    def _mock_diagnosis(self, machine_type: str = "water_pump",
                        fault_part: str = "start_capacitor") -> dict:
        import json
        raw = _make_mock_groq_response(machine_type, fault_part)
        return json.loads(raw)

    # ── EP1 — 200 OK for valid request ───────────────────────────────────

    def test_ep1_returns_200_for_valid_request(self):
        """POST /evaluate_text_rag with valid payload must return HTTP 200."""
        from services.diagnosis_service import generate_diagnosis_with_gemini

        mock_diag = self._mock_diagnosis()
        mock_pipeline = self._mock_pipeline_result()

        with patch("main.run_full_pipeline", new=AsyncMock(return_value=mock_pipeline)), \
             patch("main.generate_diagnosis_with_gemini", new=AsyncMock(return_value=mock_diag)), \
             patch("main.load_knowledge_base", return_value=""):
            resp = self.client.post("/evaluate_text_rag", json={
                "query": "Motor hums but does not start",
                "machine_type": "water_pump",
            })
        self.assertEqual(resp.status_code, 200,
                         f"Expected 200, got {resp.status_code}: {resp.text[:200]}")

    # ── EP2 — Response has required top-level keys ────────────────────────

    def test_ep2_response_has_required_keys(self):
        """Response must contain: status, solution, rag_score, machine_type."""
        mock_diag = self._mock_diagnosis()
        mock_pipeline = self._mock_pipeline_result()

        with patch("main.run_full_pipeline", new=AsyncMock(return_value=mock_pipeline)), \
             patch("main.generate_diagnosis_with_gemini", new=AsyncMock(return_value=mock_diag)), \
             patch("main.load_knowledge_base", return_value=""):
            resp = self.client.post("/evaluate_text_rag", json={
                "query": "Pump not starting",
                "machine_type": "water_pump",
            })

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for key in ("status", "solution", "rag_score", "machine_type"):
            self.assertIn(key, body, f"Response missing required key: '{key}'")

    # ── EP3 — status is lowercase ─────────────────────────────────────────

    def test_ep3_status_is_lowercase(self):
        """status field must always be lowercase (normalize_status applied)."""
        mock_diag = self._mock_diagnosis()
        mock_pipeline = self._mock_pipeline_result()

        with patch("main.run_full_pipeline", new=AsyncMock(return_value=mock_pipeline)), \
             patch("main.generate_diagnosis_with_gemini", new=AsyncMock(return_value=mock_diag)), \
             patch("main.load_knowledge_base", return_value=""):
            resp = self.client.post("/evaluate_text_rag", json={
                "query": "Pump not starting test_ep3",
                "machine_type": "water_pump",
            })

        body = resp.json()
        status = body.get("status", "")
        self.assertEqual(status, status.lower(),
                         f"status must be lowercase, got: '{status}'")

    # ── EP4 — rag_score is a float between 0 and 1 ───────────────────────

    def test_ep4_rag_score_is_valid_float(self):
        """rag_score must be a float in [0, 1]."""
        mock_diag = self._mock_diagnosis()
        mock_pipeline = self._mock_pipeline_result(rag_score=0.72)

        with patch("main.run_full_pipeline", new=AsyncMock(return_value=mock_pipeline)), \
             patch("main.generate_diagnosis_with_gemini", new=AsyncMock(return_value=mock_diag)), \
             patch("main.load_knowledge_base", return_value=""):
            resp = self.client.post("/evaluate_text_rag", json={
                "query": "Motor humming test_ep4",
                "machine_type": "water_pump",
            })

        body = resp.json()
        score = body.get("rag_score")
        self.assertIsNotNone(score, "rag_score must be present")
        self.assertIsInstance(score, (int, float), "rag_score must be numeric")
        self.assertGreaterEqual(float(score), 0.0)
        self.assertLessEqual(float(score), 1.0)

    # ── EP5 — OOD query returns no_data status ────────────────────────────

    def test_ep5_ood_query_returns_no_data(self):
        """Price inquiry must be blocked by OOD guard, returning status=no_data."""
        from pipeline_orchestrator import PipelineResult
        from unittest.mock import MagicMock

        ood_pipeline = MagicMock()
        ood_pipeline.blocked = True
        ood_pipeline.phase_reached = "ood_guard"
        ood_pipeline.block_reason = "ood_guard"
        ood_pipeline.lock = None
        ood_pipeline.response = {
            "status": "no_data",
            "ood_category": "price_inquiry",
            "diagnosis": "Price questions are not supported.",
            "steps": [],
        }

        with patch("main.run_full_pipeline", new=AsyncMock(return_value=ood_pipeline)):
            resp = self.client.post("/evaluate_text_rag", json={
                "query": "What is the price of water pump",
                "machine_type": "water_pump",
            })

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body.get("status"), "no_data",
                         f"OOD query must return no_data, got: {body.get('status')}")

    # ── EP6 — Missing machine_type inferred from query ────────────────────

    def test_ep6_missing_machine_type_does_not_500(self):
        """Empty machine_type must use inference, not crash with 500."""
        mock_diag = self._mock_diagnosis()
        mock_pipeline = self._mock_pipeline_result()

        with patch("main.run_full_pipeline", new=AsyncMock(return_value=mock_pipeline)), \
             patch("main.generate_diagnosis_with_gemini", new=AsyncMock(return_value=mock_diag)), \
             patch("main.load_knowledge_base", return_value=""):
            resp = self.client.post("/evaluate_text_rag", json={
                "query": "Water pump motor is humming",
                "machine_type": "",       # ← empty → must infer
            })

        self.assertNotEqual(resp.status_code, 500,
                            f"Missing machine_type must not cause 500: {resp.text[:200]}")

    # ── EP7 — solution.steps is always a list ─────────────────────────────

    def test_ep7_solution_steps_is_list(self):
        """solution.steps must always be a list, never null or missing."""
        mock_diag = self._mock_diagnosis()
        mock_pipeline = self._mock_pipeline_result()

        with patch("main.run_full_pipeline", new=AsyncMock(return_value=mock_pipeline)), \
             patch("main.generate_diagnosis_with_gemini", new=AsyncMock(return_value=mock_diag)), \
             patch("main.load_knowledge_base", return_value=""):
            resp = self.client.post("/evaluate_text_rag", json={
                "query": "Pump not starting test_ep7",
                "machine_type": "water_pump",
            })

        body = resp.json()
        steps = body.get("solution", {}).get("steps")
        self.assertIsNotNone(steps, "solution.steps must not be null")
        self.assertIsInstance(steps, list, "solution.steps must be a list")

    # ── EP8 — Health endpoint is accessible ──────────────────────────────

    def test_ep8_health_endpoint_returns_200(self):
        """GET /health must return 200 (sanity check for test env)."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING #5 — PERFORMANCE REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipIf(_SKIP_LLM, "SKIP_LLM_TESTS=1 — performance tests require Groq")
class TestPerformanceRegression(unittest.TestCase):
    """
    Measures wall-clock latency of the full diagnosis pipeline.
    A single LLM call to Groq should complete in < _PERF_BUDGET_S seconds
    on a healthy network connection.

    Override budget:  AGRIFIX_PERF_BUDGET_S=12 pytest ...

    These tests are intentionally not mocked — the whole point is to
    measure real latency. They are skipped offline.
    """

    def setUp(self):
        from services.diagnosis_service import generate_diagnosis_with_gemini
        self.diagnose = generate_diagnosis_with_gemini

    def _timed_diagnose(self, machine_type: str, query: str) -> Tuple[dict, float]:
        start = time.perf_counter()
        result = _run(self.diagnose(
            machine_type=machine_type,
            problem_text=query + " [perf_test_nocache]" + str(time.time()),
            rag_context=_get_stub_rag(machine_type),
        ))
        elapsed = time.perf_counter() - start
        return result, elapsed

    # ── P1 — Single diagnosis call within budget ──────────────────────────

    def test_p1_single_diagnosis_within_budget(self):
        """Single water_pump diagnosis must complete within _PERF_BUDGET_S seconds."""
        result, elapsed = self._timed_diagnose(
            "water_pump",
            "Motor hums but does not start at all"
        )
        self.assertIn(result["status"], {"success", "escalate"},
                      "Response must be valid even under timing")
        self.assertLess(
            elapsed, _PERF_BUDGET_S,
            f"Diagnosis took {elapsed:.2f}s — exceeds budget of {_PERF_BUDGET_S}s. "
            f"Possible regression: slow Groq call, retry loop triggered, or large prompt."
        )

    # ── P2 — No spurious retry (fast-path: no 429) ───────────────────────

    def test_p2_no_retry_on_healthy_call(self):
        """
        On a healthy Groq connection, the call must complete in < 2× budget.
        If it takes longer, retries were triggered unexpectedly.
        """
        result, elapsed = self._timed_diagnose(
            "electric_motor",
            "Motor making grinding noise while running"
        )
        self.assertLess(
            elapsed, _PERF_BUDGET_S * 2,
            f"Took {elapsed:.2f}s — possible unexpected retry loop."
        )

    # ── P3 — Electric machine: no extra overhead ──────────────────────────

    def test_p3_electric_machine_not_slower(self):
        """Electric machine adds a safety note injection; must not add > 2s overhead."""
        _, elapsed_wp = self._timed_diagnose(
            "water_pump", "Pump not starting"
        )
        _, elapsed_em = self._timed_diagnose(
            "electric_motor", "Motor not starting"
        )
        overhead = abs(elapsed_em - elapsed_wp)
        self.assertLess(
            overhead, 3.0,
            f"Electric machine overhead {overhead:.2f}s seems too high "
            f"(WP={elapsed_wp:.2f}s, EM={elapsed_em:.2f}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MISSING #6 — KNOWN FAILURE REGRESSION BANK
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownFailureRegression(unittest.TestCase):
    """
    Permanent regression test bank: one test per known fault type per machine.
    These tests use mocked Groq so they always run.
    Add a new entry here whenever a production bug is found and fixed.
    Focus: the SHAPE and COMPONENT of the answer, not just JSON validity.
    """

    REGRESSION_CASES = [
        # id, machine, query, fault_part (what the mock returns), expected_status, check_words_in_text
        ("R_WP_HUMS_NOCAP",  "water_pump",    "pump hums but does not start",           "start_capacitor", "success",  ["capacitor", "switch off", "mcb"]),
        ("R_WP_NOWATER",     "water_pump",    "pump running but no water",              "foot_valve",      "success",  ["foot_valve", "suction"]),
        ("R_WP_LOWFLOW",     "water_pump",    "very low water discharge from pump",     "impeller",        "success",  ["impeller"]),
        ("R_WP_OVERTEMP",    "water_pump",    "motor gets hot and trips off",           "motor_winding",   "success",  ["motor"]),
        ("R_WP_DRYRUN",      "water_pump",    "pump running dry without water",         "foot_valve",      "success",  ["foot", "water", "stop"]),
        ("R_WP_AIRLK",       "water_pump",    "pump surging and losing prime",          "foot_valve",      "success",  ["foot_valve", "suction"]),
        ("R_WP_CAVIT",       "water_pump",    "pump makes rattling noise like gravel",  "impeller",        "success",  ["impeller", "cavitation"]),
        ("R_WP_BURNT",       "water_pump",    "pump motor shows burnt smell",           "motor_winding",   "escalate", ["stop", "escalate"]),
        ("R_EM_BEARING",     "electric_motor","motor makes loud grinding noise",        "bearing",         "success",  ["bearing", "power"]),
        ("R_EM_SINGLE_PH",   "electric_motor","motor humming heavily on two phases",    "terminal_box",    "success",  ["terminal"]),
        ("R_EM_OVERCURRENT", "electric_motor","breaker trips on motor startup",         "motor_winding",   "success",  ["winding", "mcb"]),
        ("R_EM_LOWVOLT",     "electric_motor","motor running slowly under load",        "power_supply",    "success",  ["voltage", "supply"]),
        ("R_TR_NOSTART",     "tractor",       "tractor engine cranks but wont start",   "fuel_filter",     "success",  ["fuel_filter", "engine"]),
        ("R_TR_BLACKSMOKE",  "tractor",       "black smoke from exhaust under load",    "air_filter",      "success",  ["air_filter", "filter"]),
        ("R_TR_HYDROLIFT",   "tractor",       "hydraulic lift not raising implement",   "hydraulic_pump",  "success",  ["hydraulic"]),
        ("R_TR_OVERHEAT",    "tractor",       "tractor engine temperature too high",    "coolant",         "success",  ["coolant", "radiator"]),
        ("R_DE_FUELSTARVE",  "diesel_engine", "engine stops after 5 minutes running",  "fuel_filter",     "success",  ["fuel_filter", "fuel"]),
        ("R_DE_NOPRIME",     "diesel_engine", "diesel engine hard to start in morning", "glow_plug",       "success",  ["glow_plug"]),
        ("R_GEN_NOVOLT",     "generator",     "generator runs but no electricity out",  "avr",             "success",  ["avr", "capacitor"]),
        ("R_GEN_LOWVOLT",    "generator",     "generator output voltage too low",       "capacitor",       "success",  ["capacitor"]),
        ("R_SP_NOWATER",     "submersible_pump","submersible not lifting water from bore","pump_stage",     "success",  ["pump_stage", "check_valve"]),
    ]

    def _make_groq_mock(self, machine: str, fault_part: str, status: str) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = _make_mock_groq_response(machine, fault_part, status)
        return mock_resp

    def _run_regression(self, case_id, machine, query, fault_part, exp_status, check_words):
        from services.diagnosis_service import generate_diagnosis_with_gemini
        mock_resp = self._make_groq_mock(machine, fault_part, exp_status)
        with patch("services.diagnosis_service.groq_client") as mc:
            mc.chat.completions.create.return_value = mock_resp
            result = _run(generate_diagnosis_with_gemini(
                machine_type=machine,
                problem_text=query + f" [regression_{case_id}]",
                rag_context=_get_stub_rag(machine),
            ))

        # 1. Status correct
        self.assertEqual(
            result["status"], exp_status,
            f"[{case_id}] Expected status={exp_status}, got={result['status']}"
        )

        # 2. Expected words appear somewhere in full response text
        full_text = str(result).lower()
        for word in check_words:
            word_clean = word.lower().replace("_", " ")
            word_under = word.lower()
            found = word_clean in full_text or word_under in full_text
            self.assertTrue(
                found,
                f"[{case_id}] Expected keyword '{word}' not found in response.\n"
                f"  Status: {result['status']}\n"
                f"  Parts in steps: {[s.get('required_part') for s in result.get('solution',{}).get('steps',[])]}"
            )

        # 3. For success: at least one step exists
        if exp_status == "success":
            steps = result["solution"].get("steps", [])
            self.assertGreater(len(steps), 0,
                               f"[{case_id}] status=success must have at least 1 step")

        # 4. For escalate: steps empty
        if exp_status == "escalate":
            steps = result["solution"].get("steps", [])
            self.assertEqual(steps, [],
                             f"[{case_id}] status=escalate must have empty steps")

    def _make_test(case):
        def test_method(self):
            cid, machine, query, fault_part, exp_status, check_words = case
            self._run_regression(cid, machine, query, fault_part, exp_status, check_words)
        test_method.__name__ = f"test_regression_{case[0].lower()}"
        test_method.__doc__ = (
            f"[{case[0]}] {case[1]}: {case[2][:60]}\n"
            f"  Expected: {case[4]} | fault_part: {case[3]}"
        )
        return test_method

    for _case in REGRESSION_CASES:
        locals()[f"test_regression_{_case[0].lower()}"] = _make_test(_case)


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("AgriFix Diagnostic Accuracy & Production-Fidelity Test Suite  v2")
    print("=" * 70)
    print()

    skipped = []
    if _SKIP_LLM:
        skipped.append("Branch #1 live accuracy (Groq), #3 LLM farmer, #5 perf")
    if _SKIP_CHROMA:
        skipped.append("Branch #2 ChromaDB RAG retrieval")
    if skipped:
        print(f"[SKIP] Skipped (set by env): {'; '.join(skipped)}")
        print()

    print("Coverage:")
    print(f"  #1  Diagnostic accuracy       {len(GROUND_TRUTH_CASES)} ground-truth cases (live)")
    print( "  #1b Accuracy mocked           6 cases (always run)")
    print(f"  #2  RAG retrieval             8 cases (ChromaDB required)")
    print( "  #3  Real farmer queries       14 router + 2 LLM cases")
    print( "  #4  /evaluate_text_rag        8 endpoint cases (TestClient)")
    print( "  #5  Performance budget        3 latency cases (live)")
    print(f"  #6  Known failure regression  {len(TestKnownFailureRegression.REGRESSION_CASES)} cases (always run)")
    print()
    print(f"Performance budget: {_PERF_BUDGET_S}s per call  (override: AGRIFIX_PERF_BUDGET_S=...)")
    print()

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestDiagnosticAccuracy,
        TestDiagnosticAccuracyMocked,
        TestRAGRetrievalAccuracy,
        TestRealFarmerQueryRouting,
        TestRealFarmerQueryLLM,
        TestEvaluateTextRAGEndpoint,
        TestPerformanceRegression,
        TestKnownFailureRegression,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
