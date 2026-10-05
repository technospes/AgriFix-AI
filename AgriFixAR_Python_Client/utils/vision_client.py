"""Vision client — Gemini 3.1 Flash Lite with key rotation (500 req/day per key)."""
import os
import asyncio
import logging
import random
from PIL import Image
import io as io_module

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

_GEMINI_KEYS = [
    k for k in [
        os.environ.get("GOOGLE_AI_API_KEY", ""),
        os.environ.get("GOOGLE_AI_API_KEY_2", ""),
        os.environ.get("GOOGLE_AI_API_KEY_3", ""),
        os.environ.get("GOOGLE_AI_API_KEY_4", ""),
    ] if k
]

if not _GEMINI_KEYS:
    raise ValueError("No GOOGLE_AI_API_KEY keys configured")

VISION_MODEL = "gemini-3.1-flash-lite"

async def vision_call(
    prompt: str,
    image_bytes: bytes,
    max_tokens: int = 400,
    temperature: float = 0.1,
) -> str:
    """Vision call with automatic Gemini key rotation."""
    from google import genai
    from google.genai import types
    
    keys = _GEMINI_KEYS.copy()
    random.shuffle(keys)
    
    image = Image.open(io_module.BytesIO(image_bytes))
    
    for key in keys:
        try:
            client = genai.Client(api_key=key)
            
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model=VISION_MODEL,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
            )
            return response.text
            
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "exceeded" in err:
                logger.warning(f"Gemini key exhausted — rotating to next")
                continue
            raise
    
    raise Exception("All Gemini API keys exhausted")