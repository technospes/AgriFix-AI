"""
tests/test_response_accuracy.py
================================
Quick Accuracy Test Suite — ~15 cases, ~15 min runtime
Tests real Groq responses with built-in delays to avoid rate limits.

Usage:
    python tests/test_response_accuracy.py
"""
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DELAY_BETWEEN_TESTS = 20  # seconds — stay under 3 RPM to avoid Groq 429
LLM_TIMEOUT = 60          # seconds per call
PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS: List[dict] = []

# ── Color helpers ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
CYAN  = "\033[96m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ️  {msg}{RESET}")


# ── Test Cases ────────────────────────────────────────────────────────────────
# Format: (id, machine_type, query, expected_status, must_contain_keywords, must_not_contain_keywords, notes)

TEST_CASES: List[Tuple[str, str, str, str, List[str], List[str], str]] = [

    # ── NORMAL CASES (English, clear symptoms) ────────────────────────────
    (
        "N01", "water_pump",
        "Water pump is not starting at all, no sound, no vibration",
        "success",
        ["mcb", "switch off", "power", "supply"],
        ["injector", "glow_plug", "pto"],
        "Classic no-start: must suggest electrical checks"
    ),
    (
        "N02", "water_pump",
        "Pump running but no water coming out of the pipe",
        "success",
        ["foot_valve", "impeller", "suction", "prime"],
        ["injector", "glow_plug"],
        "No discharge: must mention foot valve or impeller"
    ),
    (
        "N03", "water_pump",
        "Water pump making loud grinding noise while running",
        "success",
        ["bearing", "impeller", "noise", "inspect"],
        ["injector", "fuel"],
        "Grinding noise: must suggest bearing or impeller inspection"
    ),
    (
        "N04", "electric_motor",
        "Motor hums loudly but shaft does not rotate",
        "success",
        ["capacitor", "start", "winding", "power"],
        ["injector", "foot_valve"],
        "Humming no-rotation: classic capacitor failure"
    ),
    (
        "N05", "submersible_pump",
        "Submersible pump tripping MCB immediately on startup",
        "success",
        ["winding", "capacitor", "insulation", "mcb"],
        ["injector", "glow_plug"],
        "MCB trip: must check winding or capacitor"
    ),

    # ── HINGLISH / HINDI CASES ────────────────────────────────────────────
    (
        "H01", "water_pump",
        "Paani ka pressure bohot kam aa raha hai pump se",
        "success",
        ["pressure", "impeller", "suction", "pipe"],
        [],
        "Hindi low-pressure: must retrieve relevant chunks"
    ),
    (
        "H02", "water_pump",
        "Pump chal raha hai lekin paani nahi aa raha",
        "success",
        ["foot_valve", "prime", "air", "suction"],
        [],
        "Hindi no-water: must suggest foot valve or priming"
    ),
    (
        "H03", "electric_motor",
        "Motor gunguna raha hai shuru nahi ho raha",
        "success",
        ["capacitor", "power", "winding"],
        [],
        "Hindi humming: must suggest electrical checks"
    ),

    # ── EDGE CASES ────────────────────────────────────────────────────────
    (
        "E01", "water_pump",
        "Pump",
        "success",
        ["water", "pump"],
        [],
        "Single word: system should still diagnose (not crash)"
    ),
    (
        "E02", "water_pump",
        "Not working",
        "clarification_needed",
        [],
        [],
        "Maximally vague: must ask for clarification"
    ),
    (
        "E03", "water_pump",
        "Water pump was working fine yesterday but today nothing happens",
        "success",
        ["power", "mcb", "switch", "check"],
        [],
        "Yesterday-fine-today-dead: should suggest basic checks"
    ),

    # ── SAFETY / ESCALATION CASES ─────────────────────────────────────────
    (
        "S01", "water_pump",
        "Sparks flying from the motor terminal box when I switch it on",
        "escalate",
        ["spark", "shock", "stop", "electric", "mcb"],
        [],
        "Sparks: must escalate with safety warning"
    ),
    (
        "S02", "electric_motor",
        "Burning smell and smoke coming from inside the motor",
        "escalate",
        ["burn", "smoke", "stop", "winding"],
        [],
        "Burning motor: must escalate immediately"
    ),

    # ── OUT-OF-DOMAIN CASES ───────────────────────────────────────────────
    (
        "O01", "water_pump",
        "Water pump price in market today",
        "no_data",
        [],
        [],
        "Price query: OOD guard must reject instantly"
    ),
    (
        "O02", "electric_motor",
        "Motor ke liye best brand kaunsa hai",
        "no_data",
        [],
        [],
        "Brand query: OOD guard must reject"
    ),
]


# ── Diagnosis Runner ──────────────────────────────────────────────────────────

async def run_one_test(case: tuple, index: int, total: int) -> dict:
    global PASS_COUNT, FAIL_COUNT

    cid, machine, query, exp_status, must_contain, must_not, notes = case

    print(f"\n{BOLD}[{index}/{total}] {cid} | {machine} | {query[:70]}{RESET}")
    info(f"Expected: {exp_status} | Notes: {notes}")

    try:
        from services.diagnosis_service import generate_diagnosis_with_gemini
        from rag import retrieve_with_confidence
        
        # Load ChromaDB directly
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )

        # Get real RAG context
        rag_context, rag_score, n_chunks = retrieve_with_confidence(
            vector_db=vector_db,
            query=query,
            machine_type=machine,
        )

        t0 = time.perf_counter()
        result = await asyncio.wait_for(
            generate_diagnosis_with_gemini(
                machine_type=machine,
                problem_text=query + f" [acc_test_{cid}]",
                rag_context=rag_context,
                language="en",
                vector_db=vector_db,
            ),
            timeout=LLM_TIMEOUT,
        )
        elapsed = time.perf_counter() - t0

        status = result.get("status", "unknown")
        steps = result.get("solution", {}).get("steps", [])
        n_steps = len(steps) if steps else 0

        # Build search corpus
        corpus = json.dumps(result).lower()
        steps_parts = [s.get("required_part", "") for s in steps] if steps else []

        # ── Checks ───────────────────────────────────────────────────
        status_ok = status == exp_status
        if status_ok:
            ok(f"Status: {status}")
        else:
            fail(f"Status: got '{status}', expected '{exp_status}'")

        missing = [kw for kw in must_contain if kw.lower() not in corpus]
        if not missing:
            ok(f"Keywords: all {len(must_contain)} found")
        else:
            fail(f"Missing keywords: {missing}")

        found_bad = [kw for kw in must_not if kw.lower() in corpus]
        if not found_bad:
            ok(f"Hallucination: none found")
        else:
            fail(f"Cross-machine hallucination: {found_bad}")

        if exp_status in ("success", "diagnose") and n_steps == 0:
            fail(f"Steps: 0 steps returned")
        elif n_steps > 0:
            ok(f"Steps: {n_steps} generated | parts: {steps_parts[:4]}")

        info(f"RAG score: {rag_score:.3f} | Latency: {elapsed:.1f}s")

        passed = status_ok and not missing and not found_bad
        if exp_status in ("success", "diagnose"):
            passed = passed and n_steps > 0

        if passed:
            PASS_COUNT += 1
            ok(f"{BOLD}PASS{RESET}")
        else:
            FAIL_COUNT += 1
            fail(f"{BOLD}FAIL{RESET}")

        return {
            "id": cid, "query": query, "status": status, "expected": exp_status,
            "passed": passed, "rag_score": rag_score, "latency_s": round(elapsed, 1),
            "steps": n_steps, "missing_keywords": missing,
            "hallucinated_parts": found_bad,
        }

    except asyncio.TimeoutError:
        FAIL_COUNT += 1
        fail(f"Timeout after {LLM_TIMEOUT}s")
        return {"id": cid, "query": query, "passed": False, "error": "timeout"}
    except Exception as exc:
        FAIL_COUNT += 1
        fail(f"Error: {exc}")
        return {"id": cid, "query": query, "passed": False, "error": str(exc)}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    global PASS_COUNT, FAIL_COUNT

    print(f"\n{CYAN}{'='*65}{RESET}")
    print(f"{CYAN}  AgriFix Response Accuracy Test Suite{RESET}")
    print(f"{CYAN}  {len(TEST_CASES)} cases | ~{len(TEST_CASES) * DELAY_BETWEEN_TESTS // 60} min runtime{RESET}")
    print(f"{CYAN}  Delay: {DELAY_BETWEEN_TESTS}s between tests | Timeout: {LLM_TIMEOUT}s{RESET}")
    print(f"{CYAN}{'='*65}{RESET}")

    t0_total = time.perf_counter()

    for i, case in enumerate(TEST_CASES):
        result = await run_one_test(case, i + 1, len(TEST_CASES))
        RESULTS.append(result)

        # Delay to avoid Groq rate limits
        if i < len(TEST_CASES) - 1:
            print(f"\n  {YELLOW}⏳ Waiting {DELAY_BETWEEN_TESTS}s (rate limit safety)...{RESET}")
            await asyncio.sleep(DELAY_BETWEEN_TESTS)

    total_time = time.perf_counter() - t0_total

    # ── Summary ───────────────────────────────────────────────────────────────
    total = PASS_COUNT + FAIL_COUNT
    pct = round(PASS_COUNT / total * 100, 1) if total else 0

    print(f"\n{CYAN}{'='*65}{RESET}")
    print(f"{CYAN}  RESULTS SUMMARY{RESET}")
    print(f"{CYAN}{'='*65}{RESET}")
    print(f"  Total:    {total}")
    print(f"  {GREEN}Passed:   {PASS_COUNT} ({pct}%){RESET}")
    print(f"  {RED}Failed:   {FAIL_COUNT}{RESET}")
    print(f"  Time:     {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{CYAN}{'='*65}{RESET}")

    # ── Per-category breakdown ────────────────────────────────────────────────
    categories = {"N": "Normal", "H": "Hinglish/Hindi", "E": "Edge", "S": "Safety", "O": "OOD"}
    for prefix, label in categories.items():
        cat_tests = [r for r in RESULTS if r["id"].startswith(prefix)]
        cat_pass = sum(1 for r in cat_tests if r["passed"])
        print(f"  {label}: {cat_pass}/{len(cat_tests)} passed")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "summary": {"total": total, "passed": PASS_COUNT, "failed": FAIL_COUNT,
                     "accuracy_pct": pct, "total_time_s": round(total_time, 1)},
        "results": RESULTS,
    }
    with open("accuracy_test_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → accuracy_test_results.json")

    return PASS_COUNT == total


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)