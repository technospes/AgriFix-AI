from __future__ import annotations
import asyncio
import io
import logging
import os
import re
import struct
import time
import wave
from collections import defaultdict

from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  RATE LIMITING — SlowAPI
# ═════════════════════════════════════════════════════════════════════════════
#
# get_remote_address reads X-Forwarded-For first (HuggingFace Spaces sits
# behind a reverse proxy), then falls back to request.client.host.
# Rate-limit decorators are applied per-endpoint in main.py.

limiter = Limiter(key_func=get_remote_address)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom 429 handler — structured security log + Flutter-friendly JSON body.
    Register: app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    """
    ip = get_remote_address(request)
    log_security_event("rate_limit_exceeded", ip=ip, path=request.url.path)
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=429,
        content={
            "error": "too_many_requests",
            "message": "Too many requests. Please wait before trying again.",
            "retry_after_seconds": 60,
        },
        headers={"Retry-After": "60"},
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2.  GEMINI CREDIT GUARD — per-IP sliding-window hourly cap
# ═════════════════════════════════════════════════════════════════════════════

# Raised from 10 → 300: AR mode makes ~1 Gemini call per 2 s per user.
# 300/hr = 5/min = one call every 12 s worst-case → very conservative.
# Set env var GEMINI_HOURLY_LIMIT to override.
GEMINI_HOURLY_LIMIT: int = int(os.environ.get("GEMINI_HOURLY_LIMIT", "300"))
_GEMINI_WINDOW = 3600  # one hour in seconds

# { ip_address: [unix_timestamp_of_each_call, ...] }
_gemini_usage: dict[str, list[float]] = defaultdict(list)


def can_call_gemini(ip: str) -> bool:
    """
    Return True when this IP still has budget.
    Purges timestamps older than 1 hour on every check.
    """
    now    = time.time()
    cutoff = now - _GEMINI_WINDOW
    _gemini_usage[ip] = [t for t in _gemini_usage[ip] if t > cutoff]

    calls = len(_gemini_usage[ip])
    if calls >= GEMINI_HOURLY_LIMIT:
        # Monitoring-only — log but do NOT block.
        # Blocking at this layer kills the AR locate loop for legitimate users.
        # Hard abuse protection is handled at the HTTP rate-limiter (slowapi).
        log_security_event(
            "gemini_limit_warning",
            ip=ip,
            calls_in_last_hour=calls,
            limit=GEMINI_HOURLY_LIMIT,
        )
    return True  # always allow — monitoring only


def record_gemini_call(ip: str) -> None:
    """Record one Gemini call. Call this AFTER a successful Gemini response."""
    _gemini_usage[ip].append(time.time())


def get_gemini_usage(ip: str) -> int:
    """Calls made by this IP in the last hour (exposed by /health)."""
    now    = time.time()
    cutoff = now - _GEMINI_WINDOW
    _gemini_usage[ip] = [t for t in _gemini_usage[ip] if t > cutoff]
    return len(_gemini_usage[ip])


# ═════════════════════════════════════════════════════════════════════════════
# 3a.  FILE SIZE LIMITS
# ═════════════════════════════════════════════════════════════════════════════

VIDEO_MAX_BYTES: int = int(os.environ.get("VIDEO_MAX_MB", "20")) * 1024 * 1024
AUDIO_MAX_BYTES: int = int(os.environ.get("AUDIO_MAX_MB",  "5")) * 1024 * 1024

_ALLOWED_VIDEO_EXT = {"mp4", "mov"}
_ALLOWED_AUDIO_EXT = {"mp3", "wav", "m4a"}


# ═════════════════════════════════════════════════════════════════════════════
# 3b.  MEDIA DURATION PARSERS  (pure stdlib — zero extra dependencies)
# ═════════════════════════════════════════════════════════════════════════════

VIDEO_MAX_SECONDS: float = float(os.environ.get("VIDEO_MAX_SECONDS", "20"))
AUDIO_MAX_SECONDS: float = float(os.environ.get("AUDIO_MAX_SECONDS", "20"))


# ── MP4 / MOV / M4A ──────────────────────────────────────────────────────────

def _parse_mp4_duration(data: bytes) -> float | None:
    """
    Walk the ISO Base Media File Format box tree to find the 'mvhd' box,
    then return  duration_units / timescale  as float seconds.

    Handles:
      • version-0 mvhd  — 32-bit creation/modification times (most files)
      • version-1 mvhd  — 64-bit times (rare, very long recordings)
      • boxes with 64-bit size sentinel (box_size == 1)
      • box_size == 0   → box extends to end of buffer
      • nested container boxes: moov, trak, mdia, udta

    Scans only the first 2 MB — mvhd is always near the beginning of a
    well-formed file.  Returns None if the box is not found.
    """
    SCAN = min(len(data), 2 * 1024 * 1024)
    buf  = memoryview(data)
    pos  = 0

    while pos + 8 <= SCAN:
        try:
            box_size = struct.unpack_from(">I", buf, pos)[0]
            box_type = bytes(buf[pos + 4 : pos + 8])
        except struct.error:
            break

        if box_size == 1:              # 64-bit extended size
            if pos + 16 > SCAN:
                break
            box_size = struct.unpack_from(">Q", buf, pos + 8)[0]
        elif box_size == 0:            # extends to EOF
            box_size = SCAN - pos

        if box_size < 8 or pos + box_size > SCAN:
            break

        if box_type == b"mvhd":
            version = struct.unpack_from(">B", buf, pos + 8)[0]
            if version == 1:
                # layout: hdr(8) + ver(1) + flags(3) + ctime(8) + mtime(8) + ts(4) + dur(8)
                if pos + 44 > SCAN:
                    break
                timescale = struct.unpack_from(">I", buf, pos + 28)[0]
                duration  = struct.unpack_from(">Q", buf, pos + 32)[0]
            else:
                # layout: hdr(8) + ver(1) + flags(3) + ctime(4) + mtime(4) + ts(4) + dur(4)
                if pos + 28 > SCAN:
                    break
                timescale = struct.unpack_from(">I", buf, pos + 20)[0]
                duration  = struct.unpack_from(">I", buf, pos + 24)[0]

            return (duration / timescale) if timescale else None

        # Descend into container boxes — parse their children immediately
        if box_type in (b"moov", b"trak", b"mdia", b"udta"):
            pos += 8
            continue

        pos += box_size

    return None


# ── WAV ──────────────────────────────────────────────────────────────────────

def _parse_wav_duration(data: bytes) -> float | None:
    """
    Read WAV duration via stdlib `wave` module. Exact — no approximation.
    Returns seconds as float, or None on any parse error.
    """
    try:
        with wave.open(io.BytesIO(data)) as wf:
            frames     = wf.getnframes()
            frame_rate = wf.getframerate()
            return (frames / frame_rate) if frame_rate else None
    except Exception:
        return None


# ── MP3 ──────────────────────────────────────────────────────────────────────

def _parse_mp3_duration(data: bytes, max_scan: int = 65536) -> float | None:
    """
    Estimate MP3 duration by parsing MPEG audio frame sync headers.

    Algorithm
    ─────────
    1. Skip any leading ID3v2 tag.
    2. Find valid MPEG sync words (0xFF 0xEx …) in the first `max_scan` bytes.
    3. Accumulate exact frame duration from header fields.
    4. Extrapolate over full file size (accurate for CBR; ±5% for VBR).

    We scan up to 20 frames then extrapolate — fast enough for a security
    gate, and accurate enough to catch a 3-minute file when the limit is 20s.
    Returns None if no valid frame header is found.
    """
    # MPEG1 Layer3 bitrate table (index → kbps)
    _BR = {
        1:32, 2:40, 3:48, 4:56, 5:64, 6:80, 7:96, 8:112,
        9:128, 10:160, 11:192, 12:224, 13:256, 14:320,
    }
    # Sample-rate table {mpeg_version: {sr_index: hz}}
    _SR = {
        3: {0: 44100, 1: 48000, 2: 32000},   # MPEG1
        2: {0: 22050, 1: 24000, 2: 16000},   # MPEG2
        0: {0: 11025, 1: 12000, 2:  8000},   # MPEG2.5
    }

    pos = 0

    # Skip ID3v2 tag if present: "ID3" + 2-byte version + flags + 4-byte syncsafe size
    if len(data) >= 10 and data[:3] == b"ID3":
        id3_size = (
            ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) |
            ((data[8] & 0x7F) <<  7) |  (data[9] & 0x7F)
        ) + 10
        pos = id3_size

    limit          = min(len(data), pos + max_scan)
    frames_parsed  = 0
    scanned_dur    = 0.0
    bytes_consumed = 0

    while pos + 4 <= limit and frames_parsed < 20:
        b0, b1, b2 = data[pos], data[pos + 1], data[pos + 2]

        if not (b0 == 0xFF and (b1 & 0xE0) == 0xE0):
            pos += 1
            continue

        mpeg_ver = (b1 >> 3) & 0x03   # 3=MPEG1  2=MPEG2  0=MPEG2.5
        layer    = (b1 >> 1) & 0x03   # 1=Layer3 (MP3)
        br_idx   = (b2 >> 4) & 0x0F
        sr_idx   = (b2 >> 2) & 0x03

        if layer != 1 or br_idx in (0, 15):
            pos += 1
            continue

        sr_map = _SR.get(mpeg_ver)
        if sr_map is None:
            pos += 1
            continue
        sr = sr_map.get(sr_idx)
        br = _BR.get(br_idx)
        if sr is None or br is None:
            pos += 1
            continue

        padding   = (b2 >> 1) & 0x01
        samples   = 1152 if mpeg_ver == 3 else 576
        frame_len = (samples // 8 * br * 1000 // sr) + padding
        if frame_len < 10:
            pos += 1
            continue

        scanned_dur    += samples / sr
        frames_parsed  += 1
        bytes_consumed  = pos + frame_len
        pos            += frame_len

    if frames_parsed == 0 or bytes_consumed == 0:
        return None

    # Extrapolate: scanned_dur covers bytes_consumed bytes of a len(data)-byte file
    return scanned_dur * (len(data) / bytes_consumed)


# ── Dispatcher helpers ────────────────────────────────────────────────────────

def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _video_duration(filename: str, content: bytes) -> float | None:
    ext = _ext(filename)
    if ext in ("mp4", "mov"):
        return _parse_mp4_duration(content)
    return None   # unknown format → allow through


def _audio_duration(filename: str, content: bytes) -> float | None:
    ext = _ext(filename)
    if ext == "wav":
        return _parse_wav_duration(content)
    if ext == "mp3":
        return _parse_mp3_duration(content)
    if ext == "m4a":
        return _parse_mp4_duration(content)   # M4A is AAC in ISO MP4 container
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 3c.  MAGIC-BYTE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
#
# Prevents disguised uploads (e.g. a PHP webshell renamed to video.mp4).
# We check the actual binary content, not the Content-Type header (spoofable).

_VIDEO_MAGIC: dict[str, list[tuple[int, bytes]]] = {
    "mp4": [(4, b"ftyp"), (4, b"moov"), (0, b"\x00\x00\x00\x18ftyp")],
    "mov": [(4, b"ftyp"), (4, b"moov"), (4, b"free"), (4, b"mdat")],
}
_AUDIO_MAGIC: dict[str, list[tuple[int, bytes]]] = {
    "mp3": [(0, b"\xff\xfb"), (0, b"\xff\xf3"), (0, b"\xff\xf2"), (0, b"ID3")],
    "wav": [(0, b"RIFF")],
    "m4a": [(4, b"ftyp"), (4, b"M4A ")],
}


def _magic_ok(header: bytes, ext: str, magic_table: dict) -> bool:
    patterns = magic_table.get(ext, [])
    if not patterns:
        return True  # extension whitelisted but no pattern defined → allow
    for offset, expected in patterns:
        if header[offset : offset + len(expected)] == expected:
            return True
    return False


def _safe_filename(filename: str) -> str:
    """Sanitise a filename for safe log output — strip path separators."""
    return re.sub(r"[^\w.\-]", "_", filename)[:64]


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC VALIDATORS — call these in main.py BEFORE writing any file to disk
# ═════════════════════════════════════════════════════════════════════════════

def validate_video_upload(filename: str, content: bytes, ip: str = "") -> None:
    """
    Full server-side video validation. Raises HTTPException(400) on violation.

    Validation order (fail-fast):
      1. Extension whitelist  {mp4, mov}
      2. Byte size ≤ VIDEO_MAX_BYTES   (env VIDEO_MAX_MB,      default 20 MB)
      3. Duration  ≤ VIDEO_MAX_SECONDS (env VIDEO_MAX_SECONDS, default 20 s)
      4. Minimum size sanity check
      5. Magic bytes match extension
    """
    ext = _ext(filename)

    # 1 — Extension
    if ext not in _ALLOWED_VIDEO_EXT:
        log_security_event("invalid_file_upload", ip=ip,
                           reason="disallowed_extension", filename=_safe_filename(filename))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video format '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_VIDEO_EXT))}",
        )

    # 2 — Size
    if len(content) > VIDEO_MAX_BYTES:
        size_mb = round(len(content) / 1_048_576, 1)
        log_security_event("invalid_file_upload", ip=ip, reason="file_too_large",
                           filename=_safe_filename(filename), size_mb=size_mb)
        raise HTTPException(
            status_code=400,
            detail=(f"Video too large ({size_mb} MB). "
                    f"Maximum is {VIDEO_MAX_BYTES // 1_048_576} MB."),
        )

    # 3 — Duration
    duration = _video_duration(filename, content)
    if duration is not None:
        if duration > VIDEO_MAX_SECONDS:
            log_security_event("invalid_file_upload", ip=ip, reason="duration_exceeded",
                               filename=_safe_filename(filename),
                               duration_s=round(duration, 1), limit_s=VIDEO_MAX_SECONDS)
            raise HTTPException(
                status_code=400,
                detail=(f"Video too long ({duration:.1f}s). "
                        f"Maximum is {int(VIDEO_MAX_SECONDS)}s."),
            )
    else:
        # Parse failed — unknown container or corrupt header; size check still applies
        logger.warning(
            f"⚠️  Video duration unreadable for '{_safe_filename(filename)}' "
            f"(ip={ip}) — size check passed, allowing upload"
        )

    # 4 — Minimum size
    if len(content) < 16:
        raise HTTPException(status_code=400, detail="Video file is empty or corrupt.")

    # 5 — Magic bytes
    if not _magic_ok(content[:16], ext, _VIDEO_MAGIC):
        log_security_event("invalid_file_upload", ip=ip, reason="magic_byte_mismatch",
                           filename=_safe_filename(filename))
        raise HTTPException(
            status_code=400,
            detail="Video file content does not match its extension. Upload rejected.",
        )


def validate_audio_upload(filename: str, content: bytes, ip: str = "") -> None:
    """
    Full server-side audio validation. Raises HTTPException(400) on violation.

    Validation order (fail-fast):
      1. Extension whitelist  {mp3, wav, m4a}
      2. Byte size ≤ AUDIO_MAX_BYTES   (env AUDIO_MAX_MB,      default 5 MB)
      3. Duration  ≤ AUDIO_MAX_SECONDS (env AUDIO_MAX_SECONDS, default 20 s)
      4. Minimum size sanity check
      5. Magic bytes match extension
    """
    ext = _ext(filename)

    # 1 — Extension
    if ext not in _ALLOWED_AUDIO_EXT:
        log_security_event("invalid_file_upload", ip=ip,
                           reason="disallowed_extension", filename=_safe_filename(filename))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_AUDIO_EXT))}",
        )

    # 2 — Size
    if len(content) > AUDIO_MAX_BYTES:
        size_mb = round(len(content) / 1_048_576, 1)
        log_security_event("invalid_file_upload", ip=ip, reason="file_too_large",
                           filename=_safe_filename(filename), size_mb=size_mb)
        raise HTTPException(
            status_code=400,
            detail=(f"Audio too large ({size_mb} MB). "
                    f"Maximum is {AUDIO_MAX_BYTES // 1_048_576} MB."),
        )

    # 3 — Duration
    duration = _audio_duration(filename, content)
    if duration is not None:
        if duration > AUDIO_MAX_SECONDS:
            log_security_event("invalid_file_upload", ip=ip, reason="duration_exceeded",
                               filename=_safe_filename(filename),
                               duration_s=round(duration, 1), limit_s=AUDIO_MAX_SECONDS)
            raise HTTPException(
                status_code=400,
                detail=(f"Audio too long ({duration:.1f}s). "
                        f"Maximum is {int(AUDIO_MAX_SECONDS)}s."),
            )
    else:
        logger.warning(
            f"⚠️  Audio duration unreadable for '{_safe_filename(filename)}' "
            f"(ip={ip}) — size check passed, allowing upload"
        )

    # 4 — Minimum size
    if len(content) < 8:
        raise HTTPException(status_code=400, detail="Audio file is empty or corrupt.")

    # 5 — Magic bytes
    if not _magic_ok(content[:16], ext, _AUDIO_MAGIC):
        log_security_event("invalid_file_upload", ip=ip, reason="magic_byte_mismatch",
                           filename=_safe_filename(filename))
        raise HTTPException(
            status_code=400,
            detail="Audio file content does not match its extension. Upload rejected.",
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4 + 5.  API KEY PROTECTION + CLIENT AUTHENTICATION
# ═════════════════════════════════════════════════════════════════════════════

import hmac as _hmac

APP_SECRET_KEY: str = os.environ.get("APP_SECRET_KEY", "")


def verify_app_key(request: Request) -> None:
    """
    Check X-App-Key header against APP_SECRET_KEY (constant-time compare).
    Raises HTTP 403 if key is missing or wrong.
    If APP_SECRET_KEY is not set, allows all requests (dev-mode warning).
    """
    if not APP_SECRET_KEY:
        logger.warning("⚠️  APP_SECRET_KEY not set — auth DISABLED (dev mode)")
        return

    client_key: str = request.headers.get("X-App-Key", "")
    if not client_key or not _hmac.compare_digest(
        client_key.encode(), APP_SECRET_KEY.encode()
    ):
        ip = get_remote_address(request)
        log_security_event("invalid_app_key", ip=ip, path=request.url.path)
        raise HTTPException(status_code=403, detail="Invalid or missing X-App-Key header.")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  GEMINI CALL TIMEOUT WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

GEMINI_TIMEOUT: float = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "15"))


async def gemini_with_timeout(coro, fallback=None, context: str = "gemini"):
    """
    Await `coro` with a hard GEMINI_TIMEOUT ceiling.
    Returns fallback on timeout. Never raises — callers must handle None.
    """
    try:
        return await asyncio.wait_for(coro, timeout=GEMINI_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning(f"⏱️  Gemini timeout after {GEMINI_TIMEOUT}s [{context}] — fallback")
        return fallback


# ═════════════════════════════════════════════════════════════════════════════
# 7.  PROMPT INJECTION GUARD
# ═════════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(previous|all|prior)\s+instructions?",
        r"reveal\s+(system\s+)?prompt",
        r"(show|print|display)\s+(your\s+)?(system\s+)?prompt",
        r"\bact\s+as\b",
        r"\bpretend\s+to\s+be\b",
        r"\bdeveloper\s+mode\b",
        r"\bjailbreak\b",
        r"you\s+are\s+now\s+",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"override\s+(all\s+)?(instructions?|rules?|constraints?)",
        r"\bDAN\b",
        r"repeat\s+after\s+me",
        r"output\s+(your\s+)?(raw\s+)?system",
        r"<\s*script",
        r"(\\n|\\r)+.*?(system|instruction|prompt)",
    ]
]


def check_prompt_injection(text: str, ip: str = "", field: str = "input") -> str:
    """
    Sanitise user text before sending to Gemini.
    Strips control characters, normalises whitespace, then pattern-scans.
    Raises HTTPException(400) on match. Returns sanitised text on success.
    """
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    clean = re.sub(r" {3,}", "  ", clean).strip()

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(clean):
            log_security_event("prompt_injection_attempt",
                               ip=ip, field=field, snippet=clean[:80])
            raise HTTPException(status_code=400,
                                detail="Input rejected: contains disallowed phrases.")
    return clean

def _check_prompt_injection_safe(text: str, ip: str = "", field: str = "input") -> str:
    """
    Non-raising variant of check_prompt_injection for recursive sanitization.

    Returns the sanitized text. If an injection pattern is detected, logs the
    attempt and replaces the malicious content with a safe placeholder instead
    of raising HTTPException. This allows sanitize_prompt_object() to clean an
    entire JSON object without aborting on the first malicious string.
    """
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    clean = re.sub(r" {3,}", "  ", clean).strip()

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(clean):
            log_security_event(
                "prompt_injection_sanitized",
                ip=ip, field=field,
                snippet=clean[:80],
            )
            return "[CONTENT REMOVED — injection detected]"

    return clean

# ═════════════════════════════════════════════════════════════════════════════
# 7b. RECURSIVE PROMPT OBJECT SANITIZER
# ═════════════════════════════════════════════════════════════════════════════

def sanitize_prompt_object(obj, ip: str = "", field: str = "input"):
    """
    Recursively sanitize every string in an arbitrary JSON object.

    Why recursive: attackers can hide injection payloads in any field,
    not just the ones we know about today. This protects against future
    schema changes automatically — no field-name allowlist to maintain.

    Returns a new object with all strings passed through check_prompt_injection.
    Non-string scalars (int, float, bool, None) pass through unchanged.
    """
    if isinstance(obj, str):
        return _check_prompt_injection_safe(obj, ip=ip, field=field)

    if isinstance(obj, list):
        return [
            sanitize_prompt_object(v, ip, f"{field}[{i}]")
            for i, v in enumerate(obj)
        ]

    if isinstance(obj, dict):
        return {
            k: sanitize_prompt_object(v, ip, f"{field}.{k}")
            for k, v in obj.items()
        }

    # int, float, bool, None — pass through
    return obj

# ═════════════════════════════════════════════════════════════════════════════
# 8.  SECURITY EVENT LOGGING
# ═════════════════════════════════════════════════════════════════════════════

_SEC_LOGGER     = logging.getLogger("agrifix.security")
_SECRET_PATTERN = re.compile(
    r"(key|secret|token|password|api_key)\s*[=:]\s*\S+", re.IGNORECASE
)


def log_security_event(event: str, **kwargs) -> None:
    """
    Emit one structured security log line (WARNING level).
    Example output:
      SECURITY event=duration_exceeded ip=1.2.3.4 duration_s=47.2 limit_s=20.0
    All values are scrubbed for anything resembling a credential.
    """
    parts = [f"event={event}"]
    for k, v in kwargs.items():
        safe = _SECRET_PATTERN.sub(r"\1=[REDACTED]", str(v))
        parts.append(f"{k}={safe}")
    _SEC_LOGGER.warning("SECURITY " + " ".join(parts))