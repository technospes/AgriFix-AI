from __future__ import annotations
import asyncio
import io
import logging
import os
import re
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_AUDIO_SIZE  = 10 * 1024 * 1024   # 10 MB
GROQ_MODEL      = "whisper-large-v3-turbo"
GEMINI_MODEL    = "gemini-2.5-flash-lite"
_GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")

# Thresholds for the "skip Gemini" decision
_SKIP_GEMINI_MIN_BUCKETS  = 1     # ≥1 symptom bucket detected → maybe skip
_SKIP_GEMINI_MIN_WORDS    = 4     # raw must have ≥4 words
_SKIP_GEMINI_MIN_ALPHA    = 0.60  # ≥60% of chars must be alphabetic
_SKIP_GEMINI_MAX_REPEAT   = 0.30  # <30% repeated words (stutter / junk)
_EXTRACTION_TIMEOUT       = 12.0

# Quality-gate thresholds
_SNR_GOOD      = 15.0   # dB — run normal pipeline
_SNR_DEGRADED  = 6.0    # dB — run heavier denoise before STT
_RMS_MIN       = -35.0  # dBFS — below this: too quiet
_RMS_MAX       = -3.0   # dBFS — above this: likely clipped
_DURATION_MIN  = 1.0    # seconds — below this: probably empty
_DURATION_MAX  = 25.0   # seconds — above this: trim to 25 s before STT


# ════════════════════════════════════════════════════════════════════════════════
# 1. SYMPTOM-BUCKET ONTOLOGY
# ════════════════════════════════════════════════════════════════════════════════
#
# A symptom bucket is a named category with a keyword list that covers:
#   • standard English terms
#   • transliterated Hindi (multiple spellings / Whisper variants)
#   • code-mixed forms (Hindi verb + English noun)
#   • local/colloquial variants from the field
#
# The ontology is used in three places:
#   a) Detecting which symptoms are present in the raw transcript
#   b) Verifying they survived into the Gemini output (validation gate)
#   c) Generating the PRESERVE_THESE instruction for the Gemini prompt

SYMPTOM_BUCKETS: dict[str, list[str]] = {
    "not_starting": [
        # English
        "won't start", "not start", "no start", "dead", "won't crank",
        "doesn't start", "fail to start", "starting problem", "wont start",
        "jammed", "stuck", "seized", "locked", "not turning", "not rotating",
        "cannot rotate", "motor jammed", "pump jammed", "shaft jammed",
        # Hindi/mixed — Whisper variants for "self nahi lagta" / "start nahi hota"
        "start nahi", "chal nahi", "start ho nahi", "chalu nahi",
        "self nahi", "self mar", "self lag", "self ghuma", "self ghoom",
        "self nahi lag", "self nahi chala", "ignition nahi",
        "band ho gaya", "band pad gaya",
    ],
    "clicking_cranking": [
        # Actual click sounds — Whisper tends to transcribe these phonetically
        "click", "tik tik", "tak tak", "tick tick", "tik-tik", "tak-tak",
        "khad khad", "thok thok", "khat khat", "khut khut",
        # Cranking-related
        "crank", "cranking", "self ghum", "motor ghum",
    ],
    "noise": [
        "bearing seized", "shaft seized", "metal grinding", "scraping", "binding",
        "awaaz", "awaaz aa", "sound", "noise", "kharkhara", "kharr",
        "ghar ghar", "ghur ghur", "khad khad", "drrr", "vibrat", "hilti",
        "rattle", "knock", "grind", "squeal", "hum", "humming",
        "bearing", "grinding noise", "rattling", "knocking",
    ],
    "no_water": [
        "pani nahi", "paani nahi", "water nahi", "no water", "pani band",
        "paani band", "pani kam", "water kam", "low flow", "no flow",
        "discharge nahi", "pump nahi kheench", "kheench nahi",
        "nahi kheench", "pani nahi aa", "paani nahi aa raha",
    ],
    "spinning_not_pumping": [
        "impeller stuck", "pump jammed",
        "ghoomta hai par", "ghoom raha par", "chal raha par pani nahi",
        "running but no water", "spinning but", "motor chal rahi par",
        "motor chalti hai par", "impeller", "par pani nahi",
    ],
    "smoke": [
        "dhuan", "dhuaan", "smoke", "kaala dhuan", "neela dhuan",
        "safed dhuan", "black smoke", "white smoke", "blue smoke",
        "exhaust smoke", "nikal raha dhuan",
    ],
    "smell": [
        "smell", "boo", "gandh", "jalne ki", "burning smell",
        "petrol smell", "diesel smell", "tel ki boo", "jalti hai",
        "jal raha", "jal rahi", "badboo",
    ],
    "overheating": [
        "garam", "hot", "overheat", "temperature", "garmi", "bahut garam",
        "boil", "steam", "bhaap",
    ],
    "leak": [
        "leak", "nikal raha", "nik raha", "tapak", "beh raha", "seep",
        "tel nikal", "pani nikal", "oil leak", "fuel leak", "tel aa raha",
        "coolant", "pani tapak",
    ],
    "power_loss": [
        "motor stuck", "motor locked", "shaft not turning", "impeller stuck",
        "power nahi", "dum nahi", "dheema", "slow", "sluggish", "weak",
        "kam power", "load nahi uth", "rpm kam", "speed kam", "bogging",
        "pulling weak",
    ],
    "stalling": [
        "band ho jaata", "ruk jaata", "stall", "shuts off", "cuts out",
        "stops itself", "khud band", "band pad jaata",
    ],
    "electrical": [
        "bijli", "current", "voltage", "fuse", "mcb", "relay trip",
        "short", "wire", "battery", "alternator", "capacitor",
    ],
}

def _derive_expected_buckets(machine_id: str) -> list[str]:
    """
    Return the set of symptom buckets that are plausible for a given machine.

    Fix 3: replaces the old hardcoded _MACHINE_EXPECTED_BUCKETS dict that only
    covered 11 machines by name. This function queries the machine registry
    using category flags (is_electric, is_tractor_attachment, has_fuel_system)
    so it works correctly for every machine currently in the registry AND any
    machine added in future — cultivator, sprayer, drip_irrigation, etc.

    Derivation logic:
      ALL machines      → noise, stalling, power_loss        (always possible)
      engine_driven     → not_starting, smoke, overheating, leak
      electric          → not_starting, electrical, overheating
      tractor_attachment→ (no independent starting/fuel — base set only)
      pump machines     → no_water, spinning_not_pumping      (id / cat check)
      irrigation        → no_water
      unknown/unregistered → base + not_starting (safe conservative set)
    """
    base = ["noise", "stalling", "power_loss"]

    # Lazy import so the module can be tested without the full project tree.
    # Falls back gracefully if the registry is not importable.
    try:
        from utils.machine_registry import (
            get_profile,
            is_electric_machine,
            is_tractor_attachment,
            get_fuel_system_parts,
        )
    except ImportError:
        try:
            # Flat project structure (HuggingFace Spaces layout)
            import importlib, sys as _sys
            _sys.path.insert(0, ".")
            _mr = importlib.import_module("machine_registry")
            get_profile           = _mr.get_profile
            is_electric_machine   = _mr.is_electric_machine
            is_tractor_attachment = _mr.is_tractor_attachment
            get_fuel_system_parts = _mr.get_fuel_system_parts
        except Exception:
            # Registry completely unavailable — return safe defaults
            return base + ["not_starting"]

    try:
        profile   = get_profile(machine_id)
        if profile is None:
            return base + ["not_starting"]

        is_electric = is_electric_machine(machine_id)
        is_attach   = is_tractor_attachment(machine_id)
        has_fuel    = bool(get_fuel_system_parts(machine_id))
        cat         = profile.category

        if is_attach:
            # Tractor attachments (rotavator, cultivator, seed_drill …) have no
            # independent engine — only mechanical failure modes apply.
            return list(base)

        if is_electric:
            return base + ["not_starting", "electrical", "overheating"]

        # Engine-driven machines
        if has_fuel:
            base = base + ["not_starting", "smoke", "overheating", "leak"]

        # Pump-specific buckets — catch both explicit pump machine_ids and
        # any future machine whose category marks it as a pump/irrigation type.
        _pump_ids = {"water_pump", "submersible_pump"}
        _pump_cats = {"pump", "irrigation"}
        if machine_id in _pump_ids or cat in _pump_cats:
            base = base + ["no_water", "spinning_not_pumping"]
        elif "irrigation" in machine_id or "drip" in machine_id:
            base = base + ["no_water"]

        # Dedup while preserving order
        return list(dict.fromkeys(base))

    except Exception:
        return base + ["not_starting"]


def _detect_symptom_buckets(text: str) -> dict[str, str]:
    """
    Scan text against the symptom ontology.
    Returns {bucket_name: matched_keyword} for every bucket that fires.

    Fix 2: matching strategy depends on keyword length/type:
      • Multi-word phrases ("won't start", "pani nahi aa"):
          bare substring — the phrase itself provides implicit word boundaries.
      • Single words ("hot", "dead", "hum", "crank", "wire", "short"):
          word-boundary regex (\b…\b) to avoid false positives such as
          "hot" matching "throttle", "dead" matching "deadline",
          "hum" matching "humming", "crank" matching "crankshaft".

    Hindi transliterations (awaaz, dhuan, garam, bijli) are always standalone
    words in Whisper output so \b works correctly for them too.
    The word boundary is defined by non-word chars (\\W) or string start/end in Python's re
    module, which handles ASCII alphanumerics — correct for the mixed
    Hindi-transliteration + English text Whisper produces.
    """
    t = text.lower()
    found: dict[str, str] = {}
    for bucket, keywords in SYMPTOM_BUCKETS.items():
        for kw in keywords:
            if " " in kw:
                # Multi-word phrase: bare substring is fine
                matched = kw in t
            else:
                # Single word: require word boundary to avoid substring FPs
                matched = bool(re.search(r'\b' + re.escape(kw) + r'\b', t))
            if matched:
                found[bucket] = kw
                break
    return found


# ════════════════════════════════════════════════════════════════════════════════
# 2. AUDIO QUALITY GATE
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class AudioQuality:
    snr_db:      float
    rms_dbfs:    float
    duration_s:  float
    is_clipped:  bool
    speech_frac: float   # fraction of 20ms frames above noise floor * 3
    grade:       str     # "good" | "degraded" | "poor" | "too_short"


def _estimate_audio_quality(wav_bytes: bytes) -> AudioQuality:
    """
    Estimate audio quality from a 16kHz mono WAV byte string.
    Uses numpy + stdlib wave only — no external audio libraries.

    SNR is estimated by comparing the energy of the loudest 20% of 20ms frames
    (presumed signal) against the quietest 20% (presumed noise floor).
    This is a conservative but reliable heuristic that works for speech.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            n_frames = wf.getnframes()
            sr       = wf.getframerate()
            ch       = wf.getnchannels()
            sw       = wf.getsampwidth()
            raw      = wf.readframes(n_frames)

        # Decode to float32 [-1, 1]
        if sw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        else:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2_147_483_648.0

        if ch > 1:
            samples = samples.reshape(-1, ch).mean(axis=1)

        duration  = len(samples) / max(sr, 1)
        rms       = float(np.sqrt(np.mean(samples ** 2)))
        rms_db    = 20.0 * np.log10(max(rms, 1e-9))
        is_clip   = bool(np.mean(np.abs(samples) > 0.99) > 0.001)

        frame_sz  = max(1, int(sr * 0.02))
        n_frames_q = len(samples) // frame_sz
        if n_frames_q < 4:
            return AudioQuality(0.0, rms_db, duration, is_clip, 0.0, "too_short")

        frame_rms_arr = np.array([
            float(np.sqrt(np.mean(samples[i * frame_sz:(i + 1) * frame_sz] ** 2)))
            for i in range(n_frames_q)
        ])
        frame_rms_sorted = np.sort(frame_rms_arr)
        n5 = max(1, n_frames_q // 5)

        noise_rms  = float(np.mean(frame_rms_sorted[:n5])) + 1e-9
        signal_rms = float(np.mean(frame_rms_sorted[-n5:])) + 1e-9
        snr_db     = 20.0 * np.log10(signal_rms / noise_rms)

        threshold    = noise_rms * 3.0
        speech_frac  = float(np.mean(frame_rms_arr > threshold))

        if duration < _DURATION_MIN:
            grade = "too_short"
        elif snr_db >= _SNR_GOOD and speech_frac >= 0.20:
            grade = "good"
        elif snr_db >= _SNR_DEGRADED and speech_frac >= 0.10:
            grade = "degraded"
        else:
            grade = "poor"

        logger.info(
            f"🔊 Audio quality: snr={snr_db:.1f}dB rms={rms_db:.1f}dBFS "
            f"dur={duration:.1f}s speech={speech_frac:.0%} clip={is_clip} → {grade}"
        )
        return AudioQuality(snr_db, rms_db, duration, is_clip, speech_frac, grade)

    except Exception as exc:
        logger.warning(f"⚠️  Audio quality estimate failed: {exc}")
        return AudioQuality(20.0, -16.0, 5.0, False, 0.5, "good")  # optimistic default


# ════════════════════════════════════════════════════════════════════════════════
# 3. AUDIO PRE-PROCESSING
# ════════════════════════════════════════════════════════════════════════════════

def _preprocess_audio(audio_path: Path, quality: AudioQuality | None = None) -> bytes:
    # Determine denoising strength BEFORE the try so it is available in the
    # finally log even if an exception occurs before the inner assignment.
    aggressive = (quality is not None and quality.grade == "poor")
    mode_label = "aggressive" if aggressive else "normal"
    nf_level   = "-35" if aggressive else "-25"
    tmp_path   = None

    try:
        # FIX: Removed the highly aggressive "silenceremove" filter. 
        # Whisper handles silence perfectly on its own.
        af_filters = [
            f"afftdn=nf={nf_level}",          # spectral noise reduction
            "loudnorm=I=-16:TP=-1.5:LRA=7",   # EBU R128 loudness target
        ]
        af_str = ",".join(af_filters)

        # Create temp file outside the subprocess call
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-t", str(_DURATION_MAX),       # cap at 25 s
            "-af", af_str,
            "-ar", "16000",                 # 16 kHz — Whisper's native rate
            "-ac", "1",                     # mono
            "-acodec", "pcm_s16le",         # 16-bit signed PCM
            tmp_path,
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=15)

        if result.returncode != 0:
            logger.warning(
                f"⚠️  ffmpeg pre-process failed (rc={result.returncode}, "
                f"mode={mode_label}) — using original file"
            )
            return audio_path.read_bytes()

        processed = Path(tmp_path).read_bytes()
        logger.info(
            f"🎛️  Audio pre-processed ({mode_label}): "
            f"{audio_path.stat().st_size // 1024} KB → {len(processed) // 1024} KB"
        )
        return processed

    except subprocess.TimeoutExpired:
        logger.warning(
            f"⚠️  ffmpeg timed out (mode={mode_label}) — using original file"
        )
        return audio_path.read_bytes()
    except Exception as exc:
        logger.warning(
            f"⚠️  Audio pre-process error (mode={mode_label}): {exc} — "
            f"using original file"
        )
        return audio_path.read_bytes()
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# 4. GROQ WHISPER STT
# ════════════════════════════════════════════════════════════════════════════════

_MIME_TO_EXT: dict[str, str] = {
    "audio/mpeg": "mp3", "audio/mp3": "mp3",
    "audio/mp4": "m4a", "audio/x-m4a": "m4a", "audio/m4a": "m4a",
    "audio/wav": "wav", "audio/wave": "wav", "audio/x-wav": "wav",
    "audio/webm": "webm", "video/mp4": "mp4", "video/webm": "webm",
}

# Whisper prompt: agricultural vocabulary hints help the model recognise
# domain-specific words it might otherwise mishear.
# Kept short — a long prompt biases the transcript and hurts accuracy.
_WHISPER_PROMPT = (
    "Farm machinery repair. "
    "Possible words: tractor, harvester, thresher, pump, motor, diesel, "
    "starter, self, battery, clutch, gear, belt, hydraulic, rotavator, "
    "click click, tak tak, tik tik, start nahi, chal nahi, awaaz, dhuan, "
    "pani nahi, garam, leak."
)


async def _transcribe_with_groq(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Send pre-processed WAV bytes to Groq whisper-large-v3-turbo.
    Returns raw transcript string, or "" on failure.

    Note: receives bytes (not a Path) because pre-processing already read
    and converted the file — avoids a redundant disk read.
    """
    try:
        from groq import AsyncGroq

        client   = AsyncGroq(api_key=_GROQ_API_KEY)

        logger.info(f"🎙️  Groq STT: {len(audio_bytes) // 1024} KB  file={filename}")

        response = await client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=GROQ_MODEL,
            prompt=_WHISPER_PROMPT,     # domain vocabulary hint
            # language not forced — auto-detect handles Hindi/Punjabi/English mix
            response_format="text",     # plain string, no JSON wrapping
        )

        raw = str(response).strip() if response else ""
        logger.info(f"🎙️  Groq raw: {raw[:140]!r}")
        return raw

    except ImportError:
        logger.error("❌ groq package not installed — pip install groq")
        return ""
    except Exception as exc:
        logger.error(f"❌ Groq STT: {exc}")
        return ""


# ════════════════════════════════════════════════════════════════════════════════
# 5. RAW TRANSCRIPT QUALITY CHECK → SKIP/CALL GEMINI DECISION
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class TranscriptScore:
    n_buckets:    int
    n_words:      int
    alpha_ratio:  float
    repeat_ratio: float
    needs_gemini: bool
    reason:       str
    confidence:   float   # Fix 5: heuristic 0.0–1.0, populated after audio grade known


def _compute_transcript_confidence(
    n_buckets:    int,
    n_words:      int,
    alpha_ratio:  float,
    repeat_ratio: float,
    audio_grade:  str,
) -> float:
    """
    Heuristic transcript confidence score — 0.0 (useless) to 1.0 (perfect).

    Fix 5: purely internal signal used to annotate pipeline output so callers
    (main.py, logging) can filter or warn on low-confidence transcriptions
    without needing to re-derive the quality metrics themselves.

    Formula (weighted sum, normalised to [0, 1]):
      bucket_score  = min(n_buckets / 3, 1.0)   — 3+ symptom buckets → max
      length_score  = min(n_words / 15, 1.0)    — 15+ words → max
      clean_score   = alpha_ratio                — higher alpha → cleaner output
      junk_score    = 1.0 - repeat_ratio         — less repetition → better
      audio_score   = grade-specific constant

    Weights:
      symptom signal (bucket_score)  × 0.40  — most diagnostic value
      text cleanliness (clean_score)  × 0.25
      junk penalty (junk_score)       × 0.20
      length adequacy (length_score)  × 0.10
      underlying audio quality        × 0.05  — small because pre-processing
                                                already compensates for noise

    Interpretation bands (approximate):
      ≥ 0.80  high confidence — diagnosis-ready, Gemini almost certainly skipped
      0.55–0.80  medium — acceptable, Gemini used if junk/bucket flags fired
      0.30–0.55  low — output may be imprecise; validation may have fallen back
      < 0.30  very low — audio was too short, silent, or heavily degraded
    """
    _audio_scores = {
        "good":      1.00,
        "degraded":  0.65,
        "poor":      0.30,
        "too_short": 0.00,
        "unknown":   0.70,
    }
    bucket_score = min(n_buckets / 3.0, 1.0)
    length_score = min(n_words  / 15.0, 1.0)
    clean_score  = alpha_ratio
    junk_score   = 1.0 - repeat_ratio
    audio_score  = _audio_scores.get(audio_grade, 0.70)

    raw = (
        bucket_score * 0.40 +
        clean_score  * 0.25 +
        junk_score   * 0.20 +
        length_score * 0.10 +
        audio_score  * 0.05
    )
    return round(min(max(raw, 0.0), 1.0), 3)


def _score_transcript(raw: str) -> TranscriptScore:
    """
    Decide whether to call Gemini or just lightly clean the raw transcript.

    Skip Gemini when:
      • ≥1 symptom bucket detected  (clear diagnostic signal present)
      • ≥4 words                    (not junk)
      • ≥60% alphabetic characters  (not garbled)
      • <30% repeated words         (no stutter / hallucination loop)

    Call Gemini when any of those conditions fail.

    This is the biggest latency win: for clear recordings from good-signal
    farmers (the majority), Gemini is skipped entirely.

    Note: confidence is initialised to 0.0 here and updated by
    transcribe_audio_full() once the audio quality grade is known.
    """
    if not raw or not raw.strip():
        return TranscriptScore(
            n_buckets=0, n_words=0, alpha_ratio=0.0, repeat_ratio=0.0,
            needs_gemini=True, reason="empty", confidence=0.0,
        )

    words        = raw.lower().split()
    n_words      = len(words)
    alpha_chars  = sum(ch.isalpha() for ch in raw)
    total_chars  = max(len(raw), 1)
    alpha_ratio  = alpha_chars / total_chars
    unique_words = len(set(words))
    repeat_ratio = 1.0 - (unique_words / max(n_words, 1))

    buckets = _detect_symptom_buckets(raw)
    n_b     = len(buckets)

    def _make(needs: bool, reason: str) -> TranscriptScore:
        return TranscriptScore(
            n_buckets=n_b, n_words=n_words,
            alpha_ratio=alpha_ratio, repeat_ratio=repeat_ratio,
            needs_gemini=needs, reason=reason, confidence=0.0,
        )
    
    if alpha_ratio < _SKIP_GEMINI_MIN_ALPHA:
        return _make(True, f"low_alpha({alpha_ratio:.0%})")

    if repeat_ratio > _SKIP_GEMINI_MAX_REPEAT:
        return _make(True, f"high_repeat({repeat_ratio:.0%})")

    if n_words < _SKIP_GEMINI_MIN_WORDS and n_b < _SKIP_GEMINI_MIN_BUCKETS:
        return _make(True, f"too_short_no_signal({n_words} words, {n_b} buckets)")

    if n_b < _SKIP_GEMINI_MIN_BUCKETS:
        return _make(True, "no_symptom_buckets")

    return _make(False, f"clear({n_b} buckets, {n_words} words, alpha={alpha_ratio:.0%})")


# ════════════════════════════════════════════════════════════════════════════════
# 6. STRUCTURED GEMINI EXTRACTION  (called only when needed)
# ════════════════════════════════════════════════════════════════════════════════

def _build_extraction_prompt(
    raw_transcript: str,
    detected_buckets: dict[str, str],
    machine_hint: str,
) -> str:
    """
    Build a context-aware structured extraction prompt.

    Key improvements over old freeform prompt:
      1. Forces structured output (JSON fields) — Gemini can't omit symptoms
         because they are explicit fields, not prose to be "summarised"
      2. Injects the detected symptom buckets as MUST_PRESERVE items
      3. Injects the CLIP machine hint so Gemini is biased toward the correct
         machine's vocabulary and is sceptical of unrelated machine words
    """
    preserve_lines = ""
    if detected_buckets:
        items = [f'  • {bucket} (raw evidence: "{kw}")'
                 for bucket, kw in detected_buckets.items()]
        preserve_lines = (
            "\nSYMPTOMS DETECTED IN RAW TRANSCRIPT — YOU MUST PRESERVE ALL OF THESE:\n"
            + "\n".join(items) + "\n"
        )

    machine_line = ""
    if machine_hint and machine_hint not in ("", "unknown"):
        machine_line = (
            f"\nMACHINE CONTEXT: Visual analysis identified this as a {machine_hint}. "
            f"Prefer vocabulary appropriate for {machine_hint} repair. "
            f"Be sceptical of symptom words that belong to a completely different "
            f"machine type unless the raw transcript strongly supports them.\n"
        )

    return f"""You are an expert in Indian farm machinery repair. A speech-to-text engine
produced this raw transcript of a farmer describing their machine problem:

RAW TRANSCRIPT:
{raw_transcript}
{preserve_lines}{machine_line}
Extract the problem description as a JSON object with EXACTLY these fields:
  "machine": the machine name in English (e.g. "tractor", "water pump")
  "symptom_en": 1 sentence in plain English describing the main symptom
  "sound_detail": any specific sound described (e.g. "clicking noise", "tak tak sound") or null
  "failure_mode": what is failing (e.g. "won't start", "no water output") or null
  "full_description": 1-2 plain English sentences combining machine + symptom + sound

STRICT RULES:
1. symptom_en MUST include every symptom bucket listed in SYMPTOMS DETECTED above.
2. sound_detail MUST be filled if any sound word appears in the raw (click, tak, tik,
   awaaz, khad khad, kharkhara, ghar ghar, etc.). Do NOT set to null if raw has a sound.
3. failure_mode MUST be filled if raw mentions starting failure (start nahi, self nahi,
   chal nahi, won't start) or output failure (pani nahi, no water, bijli nahi).
4. full_description must include all of: machine + symptom_en + sound_detail
   (if any) + failure_mode (if any). Do NOT drop any of these.
5. Do NOT invent symptoms not present in raw.
6. Return ONLY valid JSON. No markdown, no explanation.

Example output:
{{
  "machine": "tractor",
  "symptom_en": "makes a rapid clicking sound and will not start",
  "sound_detail": "clicking sound",
  "failure_mode": "won't start",
  "full_description": "The tractor makes a rapid clicking sound and will not start."
}}"""


async def _extract_with_gemini(
    raw_transcript: str,
    detected_buckets: dict[str, str],
    machine_hint: str,
) -> str:
    """
    Structured Gemini extraction — called only when raw transcript quality
    requires LLM interpretation. Returns full_description string.

    Uses forced JSON output so Gemini cannot silently drop fields.
    Falls back to light_clean_raw(raw_transcript) on parse failure.
    """
    if not raw_transcript.strip():
        return "farm machine problem"

    try:
        from utils.helpers import sanitize_json_text
        import json as _json

        prompt   = _build_extraction_prompt(raw_transcript, detected_buckets, machine_hint)
        model    = genai.GenerativeModel(GEMINI_MODEL)
        response = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: model.generate_content(prompt)
            ),
            timeout=_EXTRACTION_TIMEOUT,
        )
        raw_json = sanitize_json_text(response.text or "")

        try:
            data = _json.loads(raw_json)
            full_desc = (data.get("full_description") or "").strip()
            if full_desc:
                # Strip "unknown machine" hallucination
                full_desc = re.sub(
                    r"\b(?:the\s+)?unknown\s+machine'?s?\b\s*",
                    "",
                    full_desc,
                    flags=re.IGNORECASE,
                ).strip()
                # Also strip leading "on the" or "in the" left behind
                full_desc = re.sub(r"^(?:on|in)\s+the\s+", "", full_desc, flags=re.IGNORECASE).strip()
                # Capitalize first letter
                if full_desc:
                    full_desc = full_desc[0].upper() + full_desc[1:]
                logger.info(f"Gemini structured: {full_desc[:120]}")
                return full_desc
        except (_json.JSONDecodeError, AttributeError):
            # JSON parse failed — fall through to raw fallback
            logger.warning(
                f"⚠️  Gemini returned non-JSON: {raw_json[:80]} — using raw"
            )

    except asyncio.TimeoutError:
        logger.warning(
            f"⏱️  Gemini structured extraction exceeded {_EXTRACTION_TIMEOUT}s — "
            f"using cleaned raw transcript instead of discarding it"
        )
        return _light_clean_raw(raw_transcript)

    except ImportError:
        # sanitize_json_text not available in test context — use stdlib
        import json as _json
        try:
            model    = genai.GenerativeModel(GEMINI_MODEL)
            prompt   = _build_extraction_prompt(raw_transcript, detected_buckets, machine_hint)
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: model.generate_content(prompt)
            )
            txt = (response.text or "").strip().lstrip("```json").rstrip("```").strip()
            data = _json.loads(txt)
            full_desc = (data.get("full_description") or "").strip()
            if full_desc:
                return full_desc
        except Exception:
            pass
    except Exception as exc:
        logger.error(f"❌ Gemini extraction: {exc}")

    return _light_clean_raw(raw_transcript)


# ════════════════════════════════════════════════════════════════════════════════
# 7. SYMPTOM-BUCKET VALIDATION GATE
# ════════════════════════════════════════════════════════════════════════════════

def _validate_output(
    raw: str,
    output: str,
    raw_buckets: dict[str, str],
    machine_hint: str,
) -> tuple[bool, str]:
    """
    Validate that the final output preserves every symptom bucket that was
    detected in the raw transcript, and has not hallucinated machine context.

    Two checks:

    A) Bucket survival — for every bucket found in raw, at least one keyword
       from that bucket must appear in the output.  This uses the same
       word-boundary strategy as _detect_symptom_buckets so the check is
       consistent with detection.

    B) Machine-context cross-check — detects the most common hallucination
       pattern: Gemini outputs a machine name that CLIP did not identify AND
       that machine's name is absent from the raw transcript.
       Built dynamically from _derive_expected_buckets so it covers ALL
       machines in the registry, not just the 3 that the old hardcoded
       _foreign_machine_words dict handled.

    Returns (is_valid: bool, reason: str).
    """
    output_lower = output.lower()
    raw_lower    = raw.lower()

    # ── A: Bucket survival check ─────────────────────────────────────────────
    for bucket, raw_kw in raw_buckets.items():
        output_kws = SYMPTOM_BUCKETS[bucket]
        # Use word-boundary matching to be consistent with detection
        survived = False
        for kw in output_kws:
            if " " in kw:
                survived = kw in output_lower
            else:
                survived = bool(re.search(r'\b' + re.escape(kw) + r'\b', output_lower))
            if survived:
                break
        if not survived:
            return False, f"bucket_lost:{bucket}(raw_evidence='{raw_kw}')"

    # ── B: Machine-context cross-check ───────────────────────────────────────
    # Skip when no CLIP hint is available.
    if not machine_hint or machine_hint in ("", "unknown"):
        return True, "ok"

    # Derive the plausible bucket set for the detected machine — registry-driven,
    # works for all machines including future additions.
    expected_buckets = set(_derive_expected_buckets(machine_hint))

    # Build a list of "foreign machine" indicator phrases: common names of
    # OTHER machine types that should NOT appear in the output unless the raw
    # transcript also contains them.
    # We generate this dynamically from a small universal alias map so there
    # is no per-machine hardcoding.
    _machine_aliases: dict[str, list[str]] = {
        "tractor":          ["tractor", "ट्रैक्टर"],
        "water_pump":       ["water pump", "pump set", "monoblock", "पानी का पंप"],
        "submersible_pump": ["submersible", "borewell", "tubewell", "पाताल पंप"],
        "harvester":        ["harvester", "combine", "हार्वेस्टर"],
        "thresher":         ["thresher", "थ्रेशर"],
        "electric_motor":   ["electric motor", "bijli motor"],
        "generator":        ["generator", "genset", "जनरेटर"],
        "power_tiller":     ["power tiller", "walking tractor"],
        "chaff_cutter":     ["chaff cutter", "toka machine", "fodder cutter"],
        "diesel_engine":    ["diesel engine", "stationary engine"],
        "rotavator":        ["rotavator", "rotary tiller"],
        "cultivator":       ["cultivator", "kultivator"],
        "sprayer":          ["sprayer", "spray machine", "dawa machine"],
        "drip_irrigation":  ["drip irrigation", "drip system", "sprinkler"],
    }

    for other_machine, aliases in _machine_aliases.items():
        if other_machine == machine_hint:
            continue  # Not a "foreign" machine — skip
        for alias in aliases:
            # If the output mentions another machine's name AND the raw
            # transcript does NOT mention it → likely a hallucination
            if alias in output_lower and alias not in raw_lower:
                return False, f"machine_hallucination:'{alias}'_not_in_raw"

    return True, "ok"


# ════════════════════════════════════════════════════════════════════════════════
# 8. FILLERS + LIGHT CLEAN
# ════════════════════════════════════════════════════════════════════════════════

_FILLERS = re.compile(
    r'\b(um+|uh+|ahem|hmm+|haan+|haan ji|ji+|bhai|yaar|'
    r'dekho|suno|achha|theek hai|matlab|toh|bas|'
    r'please|help me|bata do|batao|kya karoon|'
    r'ek baar|dobara|sun|arre|oye)\b',
    re.IGNORECASE,
)


def _light_clean_raw(raw: str) -> str:
    """
    Minimal cleaning for use when Gemini is skipped or validation fails.
    Removes filler words and excessive whitespace.
    Preserves all symptom-carrying words exactly as Whisper produced them.
    """
    cleaned = _FILLERS.sub(" ", raw)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned if len(cleaned) > 15 else raw


# ════════════════════════════════════════════════════════════════════════════════
# 9. GEMINI AUDIO FALLBACK  (original path — used when GROQ_API_KEY absent)
# ════════════════════════════════════════════════════════════════════════════════

async def _transcribe_with_gemini_audio(audio_path: Path) -> str:
    """
    Original Gemini audio transcription. Kept exactly to ensure zero
    behaviour change when GROQ_API_KEY is not set.
    """
    logger.info(f"🎤 Gemini audio fallback: {audio_path}")
    try:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(audio_path))
        mime_type     = mime_type or "audio/mp4"

        audio_file = genai.upload_file(str(audio_path), mime_type=mime_type)
        model      = genai.GenerativeModel(GEMINI_MODEL)

        prompt = """You are an expert assistant who understands how Indian farmers speak about problems with their farm machinery and equipment.

The farmer may speak Hindi, Punjabi, Bhojpuri, Haryanvi, Marathi, Gujarati, Tamil, Telugu, Kannada, Odia, Bengali — or a mix. They describe problems using everyday physical words, not technical terms.

Your job: Listen carefully and return a clear, plain English description of the problem. The machine could be any farm equipment — tractor, harvester, thresher, water pump, submersible pump, electric motor, power tiller, rotavator, chaff cutter, generator, or diesel engine.

COMMON PHRASES FOR ALL MACHINES:

STARTING / POWER PROBLEMS (any machine):
- "chal nahi raha / start nahi ho raha" = machine won't start
- "band ho jaata hai / ruk jaata hai" = machine stops by itself
- "bijli nahi aa rahi / self nahi lag raha" = no electricity / electric start failing
- "dheema chal raha hai / power nahi hai / dum nahi hai" = running slow, losing power

TRACTOR / ENGINE:
- "clutch kaam nahi kar raha" = clutch not working
- "gear nahi lag raha / gear phasna" = gear stuck
- "zyada dhuan / kaala dhuan" = excessive smoke
- "tel nikal raha" = oil or fuel leaking
- "pani nikal raha" = coolant leaking
- "awaaz aa rahi / khad khad / thok thok" = knocking/rattling noise
- "belt toot gayi / belt slip kar rahi" = belt broke or slipping
- "engine garam ho raha / overheat" = overheating
- "steering tight / ghoomti nahi" = stiff steering
- "hydraulic nahi uth raha" = hydraulic lift not working

HARVESTER / THRESHER:
- "machine jam ho gayi / anaj phas gayi" = crop jam
- "anaj nahi nikal raha / bhusa mein anaj" = grain not separating
- "anaj kat raha / toot raha" = grain being cracked
- "chalni band / jali jam" = sieve blocked
- "drum jam / cylinder jam" = threshing drum jammed

PUMPS (water pump / submersible):
- "pani nahi aa raha / pani band" = no water output
- "pani kam aa raha / pressure kam" = low flow or pressure
- "motor nahi chali / motor jal gayi" = motor won't start or burned
- "motor gunjti hai par nahi chalti" = hums but won't spin
- "pani ulta aa raha" = water falls back (check valve problem)
- "fuse ud gayi / MCB trip / relay trip" = fuse blown or relay tripping

CHAFF CUTTER:
- "toka nahi kat raha / blade dull ho gayi" = not cutting cleanly
- "toka jam ho gaya" = jammed
- "machine baar baar band hoti hai" = keeps stopping

GENERATOR:
- "bijli nahi aa rahi / current nahi hai" = no electrical output
- "voltage kam / voltage upar neeche" = low or unstable voltage
- "generator band ho jaata hai" = starts then stops

GENERAL:
- "awaaz bhaari / kharkhara" = grinding noise
- "vibration zyada / bahut hilti" = excessive vibration
- "bearing kharaab / ghar ghar" = bearing noise
- "belt dhili / belt nikal jaati" = belt loose or coming off

INSTRUCTIONS:
1. Identify which machine the farmer is talking about.
2. Understand the problem even if speech is unclear or code-switched.
3. Return 1-2 plain English sentences: (a) which machine, (b) what the problem is.
4. Use simple words — no technical jargon.
5. Do NOT include original speech, translations, brackets, or labels.

Return ONLY the plain English problem description. Nothing else."""

        response      = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.generate_content([audio_file, prompt])
        )
        transcription = (response.text or "").strip()
        logger.info(f"📝 Gemini audio: {transcription[:120]}...")

        try:
            genai.delete_file(audio_file.name)
        except Exception:
            pass

        return transcription if transcription else "farm machine problem"

    except Exception as exc:
        logger.error(f"❌ Gemini audio fallback error: {exc}")
        return "farm machine problem"


# ════════════════════════════════════════════════════════════════════════════════
# 10. PUBLIC API
# ════════════════════════════════════════════════════════════════════════════════

async def transcribe_audio_full(
    audio_path: Path,
    machine_hint: str = "",
) -> dict[str, str]:
    """
    Full pipeline. Returns:
        {
            "raw_transcript":     "<exact Whisper output>",
            "normalized_problem": "<final diagnosis-ready English>",
            "quality_grade":      "good" | "degraded" | "poor" | "too_short",
            "gemini_used":        "true" | "false",
            "skip_reason":        "<why Gemini was skipped, or empty>",
        }

    machine_hint: canonical machine_id from CLIP detection (e.g. "tractor").
    Pass "" if not available yet — pipeline still works, just without
    the context cross-check.

    Fallback chain:
        GROQ_API_KEY present + Groq succeeds  → full pipeline (stages 1-7)
        GROQ_API_KEY present + Groq fails     → Gemini audio fallback
        GROQ_API_KEY absent                   → Gemini audio fallback
    """
    logger.info(
        f"🎤 transcribe_audio_full: {audio_path.name}  "
        f"machine_hint={machine_hint!r}"
    )

    if not _GROQ_API_KEY:
        logger.info("ℹ️  GROQ_API_KEY not set — Gemini audio fallback")
        result = await _transcribe_with_gemini_audio(audio_path)
        return {
            "raw_transcript":        result,
            "normalized_problem":    result,
            "quality_grade":         "unknown",
            "gemini_used":           "true",
            "skip_reason":           "no_groq_key",
            "transcript_confidence": 0.5,
        }

    # ── Stage 1: Pre-process audio ────────────────────────────────────────────
    # First pass: convert to 16kHz WAV for quality estimation + STT.
    # On ffmpeg failure (binary not on PATH, e.g. Windows dev machine),
    # _preprocess_audio returns the original file bytes unchanged.
    # Detect this by checking for the RIFF WAV magic bytes at offset 0.
    raw_bytes_for_quality = _preprocess_audio(audio_path, quality=None)
    _ffmpeg_ok = raw_bytes_for_quality[:4] == b"RIFF"

    if not _ffmpeg_ok:
        # ffmpeg was not available — send the original file to Groq
        # with its real extension so Groq picks the right decoder.
        # The WAV quality gate is skipped because it only works on PCM data.
        logger.warning(
            "⚠️  ffmpeg produced no WAV output — "
            "sending original file to Groq directly (quality gate skipped)"
        )
        _orig_ext      = (audio_path.suffix or ".m4a").lstrip(".").lower()
        _orig_filename = f"audio.{_orig_ext}"
        _orig_bytes    = audio_path.read_bytes()
        quality        = AudioQuality(
            snr_db=15.0, rms_dbfs=-16.0, duration_s=10.0,
            is_clipped=False, speech_frac=0.5, grade="unknown",
        )
        raw = await _transcribe_with_groq(_orig_bytes, filename=_orig_filename)

        if not raw:
            logger.warning("⚠️  Groq STT empty (no ffmpeg) — Gemini audio fallback")
            result = await _transcribe_with_gemini_audio(audio_path)
            return {
                "raw_transcript":        result,
                "normalized_problem":    result,
                "quality_grade":         "unknown",
                "gemini_used":           "true",
                "skip_reason":           "groq_failed_no_ffmpeg",
                "transcript_confidence": 0.5,
            }

    else:
        # ── Stage 2: Audio quality gate ───────────────────────────────────────
        quality = _estimate_audio_quality(raw_bytes_for_quality)

        if quality.grade == "too_short":
            logger.warning("⚠️  Audio too short — returning fallback")
            return {
                "raw_transcript":        "",
                "normalized_problem":    "farm machine problem",
                "quality_grade":         "too_short",
                "gemini_used":           "false",
                "skip_reason":           "audio_too_short",
                "transcript_confidence": 0.0,
            }

        # Re-process with quality-aware settings (aggressive denoising if poor)
        processed_bytes = (
            _preprocess_audio(audio_path, quality=quality)
            if quality.grade == "poor"
            else raw_bytes_for_quality
        )

        # ── Stage 3: Groq Whisper STT ─────────────────────────────────────────
        raw = await _transcribe_with_groq(processed_bytes, filename="audio.wav")

    if not raw:
        logger.warning("⚠️  Groq STT empty — Gemini audio fallback")
        result = await _transcribe_with_gemini_audio(audio_path)
        return {
            "raw_transcript":        result,
            "normalized_problem":    result,
            "quality_grade":         quality.grade,
            "gemini_used":           "true",
            "skip_reason":           "groq_failed",
            "transcript_confidence": 0.5,
        }

    # ── Stage 4: Raw transcript quality score → skip/call Gemini ─────────────
    score   = _score_transcript(raw)
    buckets = _detect_symptom_buckets(raw)

    # Fix 5: compute real confidence now that audio grade is known
    score.confidence = _compute_transcript_confidence(
        n_buckets    = score.n_buckets,
        n_words      = score.n_words,
        alpha_ratio  = score.alpha_ratio,
        repeat_ratio = score.repeat_ratio,
        audio_grade  = quality.grade,
    )

    logger.info(
        f"📊 Transcript score: buckets={score.n_buckets} words={score.n_words} "
        f"alpha={score.alpha_ratio:.0%} repeat={score.repeat_ratio:.0%} "
        f"needs_gemini={score.needs_gemini} reason={score.reason} "
        f"confidence={score.confidence:.3f}"
    )

    # ── Stage 5: Normalise ────────────────────────────────────────────────────
    if not score.needs_gemini:
        # Skip Gemini — raw transcript is already clear enough
        normalised = _light_clean_raw(raw)
        gemini_used = False
        skip_reason = score.reason
        logger.info(f"⚡ Gemini skipped ({skip_reason}) — using cleaned raw")
    else:
        # Call Gemini structured extraction
        normalised  = await _extract_with_gemini(raw, buckets, machine_hint)
        gemini_used = True
        skip_reason = ""

    # ── Stage 6: Symptom-bucket validation ───────────────────────────────────
    is_valid, val_reason = _validate_output(raw, normalised, buckets, machine_hint)

    if not is_valid:
        logger.warning(
            f"⚠️  Validation failed ({val_reason}) — "
            f"falling back to cleaned raw"
        )
        normalised = _light_clean_raw(raw)

    logger.info(
        f"✅ Transcription complete | valid={is_valid}\n"
        f"   raw:        {raw[:100]!r}\n"
        f"   normalised: {normalised[:100]!r}"
    )

    return {
        "raw_transcript":       raw,
        "normalized_problem":   normalised,
        "quality_grade":        quality.grade,
        "gemini_used":          str(gemini_used).lower(),
        "skip_reason":          skip_reason,
        "transcript_confidence": score.confidence,  # Fix 5: heuristic 0.0–1.0
    }


async def transcribe_audio_with_gemini(audio_path: Path) -> str:
    """
    Drop-in replacement for original function — API unchanged.
    All callers in main.py continue to work without modification.
    Returns normalized_problem string only.
    machine_hint is not passed here; main.py uses transcribe_audio_full()
    directly when it wants to pass the CLIP hint.
    """
    result = await transcribe_audio_full(audio_path)
    return result["normalized_problem"]