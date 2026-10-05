"""
utils/image_utils.py
Shared image resizing — single source of truth for all services.

Resizes images while preserving aspect ratio. Never enlarges.
Intentionally converts output to JPEG for Gemini token efficiency.
Orientation is corrected via EXIF metadata. Thread-safe.
Returns original bytes on failure (never raises).
"""

from __future__ import annotations
import asyncio
import io
import logging

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


async def resize_image(
    image_bytes: bytes,
    *,
    max_dim: int,
    quality: int = 85,
) -> bytes:
    """
    Resize an image so its largest dimension does not exceed max_dim pixels.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)
        max_dim: Maximum width or height in pixels. Required — no default.
        quality: JPEG output quality, clamped to 1–95. Default 85.

    Returns:
        JPEG bytes at or below max_dim, or the original bytes on error.

    Thread-safe: runs PIL operations in a thread via asyncio.to_thread.
    """
    if not isinstance(image_bytes, (bytes, bytearray)):
        logger.warning("resize_image: expected bytes, got %s", type(image_bytes).__name__)
        return image_bytes

    quality = max(1, min(quality, 95))

    try:
        return await asyncio.to_thread(
            _resize_sync,
            image_bytes,
            max_dim,
            quality,
        )
    except Exception as exc:
        logger.warning(
            "Image resize failed (max_dim=%d, quality=%d): %s",
            max_dim, quality, exc,
        )
        return image_bytes


def _resize_sync(image_bytes: bytes, max_dim: int, quality: int) -> bytes:
    """Synchronous PIL resize — called via asyncio.to_thread."""

    with Image.open(io.BytesIO(image_bytes)) as img:
        # Correct orientation from EXIF (portrait photos otherwise appear sideways)
        img = ImageOps.exif_transpose(img)

        # Normalize mode for JPEG output — RGBA/P images can fail or distort
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        if img.width <= max_dim and img.height <= max_dim:
            # Already within bounds — still re-encode for consistent JPEG output
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()

        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()