"""
utils/text_utils.py
Shared tokenization utilities — single source of truth for stop words and
tokenization across the retrieval pipeline.

Used by:
  • rag.py          — _tokenize(), _STOP_WORDS
  • mmr_dedup.py    — _tokenize(), _STOP_WORDS

Behavior intentionally mirrors the previous implementations to avoid
changing retrieval quality. Stop words are a frozen set (immutable) to
prevent accidental mutation at runtime.
"""

from __future__ import annotations
import re
from typing import FrozenSet, List

# ── Canonical stop words (English + Hindi) ────────────────────────────────────
# Previously duplicated in rag.py and mmr_dedup.py with slight divergence
# (mmr_dedup.py was missing "phir","ab","ya","hoga"). Now unified and immutable.

STOP_WORDS: FrozenSet[str] = frozenset({
    # English
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
    "from","is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall","can",
    "not","no","nor","so","yet","both","either","each","more","most","other",
    "some","such","than","that","this","these","those","it","its","my","your",
    "his","her","our","their","i","we","you","he","she","they","what","which",
    "who","when","where","how","if","as","up","out","about","into","through",
    "during","before","after","above","below","between","there","here","then",
    "any","all",
    # Hindi / Hinglish
    "hai","hain","ka","ke","ki","ko","se","mein","par","aur","ek","yeh","woh",
    "kya","kab","kaise","nahi","nahin","bhi","toh","jo","jab","agar","lekin",
    "phir","ab","ya","hoga",
})

# Pre-compiled regex for tokenization — avoids re-compilation on every call.
_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase word tokens, removing stop words and
    single-character tokens.

    Args:
        text: Raw text string (English, Hindi, or Hinglish).

    Returns:
        List of meaningful word tokens.
    """
    raw = _TOKEN_RE.findall(text.lower())
    return [t for t in raw if len(t) > 1 and t not in STOP_WORDS]


__all__ = [
    "STOP_WORDS",
    "tokenize",
]