from __future__ import annotations
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_DEVICE     = os.environ.get("CLIP_DEVICE", "cpu")
_MODEL_NAME      = os.environ.get("CLIP_MODEL_NAME", "MobileCLIP-S1")
_MODEL_PRETRAINED = os.environ.get("CLIP_PRETRAINED",  "datacompdr")

# ── Classification thresholds (cosine similarity after softmax) ───────────────
# How these were chosen:
#   CLIP ViT-B/32 zero-shot on farm machinery:
#     Clear, well-lit machine photo  → 0.65–0.85
#     Partial view / background clutter → 0.45–0.60
#     Wrong machine or non-machinery  → 0.10–0.30  (at T=50)
#   0.50 = "confident enough to skip Gemini"           (at T=50)
#   0.35 = "has a weak signal, worth including in fusion" (at T=50)
#
# These thresholds are calibrated for softmax temperature T=50.
# Production data (gap=0.026 → top=0.649) confirms clear matches
# sit well above 0.50, and ambiguous cases (gap≤0.008) fall below 0.35.
_CLIP_STRONG     = 0.50   # trust CLIP alone above this  (was 0.36 @ T=30)
_CLIP_WEAK       = 0.35   # include in fusion above this (unchanged — still valid @ T=50)
_AUDIO_STRONG    = 0.70   # trust audio alone above this
_AUDIO_WEAK      = 0.40   # include in fusion above this
_FUSED_MIN       = 0.50   # fused result above this → skip Gemini

# ── Frame quality thresholds ──────────────────────────────────────────────────
_BLUR_MIN        = 80.0   # Laplacian variance — below = blurry, reject
_BRIGHT_MIN      = 40     # mean pixel value — below = too dark
_BRIGHT_MAX      = 220    # mean pixel value — above = overexposed
_clip_model       = None
_clip_preprocess  = None   # torchvision transform pipeline (replaces CLIPProcessor)
_clip_tokenizer   = None   # open_clip tokenizer
_text_embeddings: Optional[dict] = None   # { machine_id: torch.Tensor }
_clip_ready       = False

_MACHINE_PROMPTS: dict[str, list[str]] = {
    "tractor": [
        "a photo of a farm tractor with large rear wheels in a field",
        "a diesel tractor with a metal bonnet and steering wheel",
        "a Mahindra or Sonalika agricultural tractor",
        "a four-wheel drive tractor used for ploughing",
    ],
    "harvester": [
        "a photo of a combine harvester cutting wheat crop",
        "a large self-propelled grain harvester in a field",
        "a combine with a wide cutting header at the front",
        "a CLAAS or John Deere combine harvester",
    ],
    "thresher": [
        "a photo of a stationary grain threshing machine",
        "a farm thresher separating wheat grain from stalks",
        "a multicrop thresher with a feed inlet and discharge chute",
        "a blue agricultural thresher powered by a tractor belt",
    ],
    "submersible_pump": [
        "a photo of a borewell submersible pump control panel",
        "a metal starter box with MCB switches for a borewell motor",
        "an electrical panel box mounted on a pole for underground pump",
        "a submersible pump starter with indicator lights and wiring",
    ],
    "water_pump": [
        "a photo of a diesel water pump set for farm irrigation",
        "a centrifugal monoblock pump with suction and delivery pipes",
        "a surface irrigation pump with rubber hoses attached",
        "a water pump engine set near an irrigation canal",
    ],
    "electric_motor": [
        "a photo of an electric motor driving farm equipment",
        "an induction motor with a belt and pulley on a metal base",
        "a three-phase electric motor with a terminal box",
        "an electric motor control panel with overload relay",
    ],
    "power_tiller": [
        "a photo of a walk-behind power tiller in a paddy field",
        "a two-wheeled diesel power tiller tilling wet soil",
        "a VST Shakti mini power tiller with rotary blades",
        "a farmer operating a walk-behind tractor with handlebars",
    ],
    "rotavator": [
        "a photo of a rotavator attached to the back of a tractor",
        "a rotary tiller implement with L-shaped blades on a tractor PTO",
        "a soil rotary cultivator being pulled through a field",
        "a tractor-mounted rotavator tilling agricultural land",
    ],
    "chaff_cutter": [
        "a photo of a chaff cutter fodder cutting machine in a farm",
        "a green fodder cutter with a feed inlet and discharge chute",
        "an electric agricultural straw cutter for animal feed",
        "a chaff cutting machine with rotating blades and a flywheel",
    ],
    "generator": [
        "a photo of a portable diesel generator on a farm",
        "a petrol generator set producing electricity outdoors",
        "a small farm genset with an engine and output sockets",
        "a generator with a fuel tank and alternator housing",
    ],
    "diesel_engine": [
        "a photo of a stationary single-cylinder diesel engine on a farm",
        "a Kirloskar or Lister diesel engine with a large flywheel",
        "a diesel engine driving a water pump by a belt",
        "a standalone diesel engine with an exhaust pipe and decompression lever",
    ],
    # ── New machines added: cultivator / sprayer / irrigation ────────────────
    "cultivator": [
        "a photo of a tractor-mounted cultivator implement in a field",
        "a soil cultivator with tines or shovels attached to a tractor",
        "a farm cultivator stirring and aerating soil between crop rows",
        "a multi-row cultivator frame being pulled by a tractor",
    ],
    "sprayer": [
        "a photo of a knapsack or back-mounted pesticide sprayer",
        "a motorised crop sprayer being used in a farm field",
        "a boom sprayer mounted on a tractor spraying pesticide",
        "a hand-pump agricultural sprayer with a pressure tank and nozzle",
    ],
    "drip_irrigation": [
        "a photo of drip irrigation pipes and emitters in a farm field",
        "a drip irrigation filter unit and pressure regulator on a farm",
        "a micro-irrigation lateral pipe with drippers installed in soil",
        "a drip irrigation control valve and fertigation tank assembly",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# 3.  AUDIO KEYWORD CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs on the transcription text produced by transcription_service.
# Zero API calls. Pure string matching. Runtime: <1ms.
#
# Each entry: (keyword, weight)
# Higher weight = more specific to that machine (e.g. "thresher" = 1.0)
# Lower weight = shared term (e.g. "engine" = 0.3)

_AUDIO_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "tractor": [
        ("tractor", 1.0), ("ट्रैक्टर", 1.0),
        ("mahindra", 0.9), ("sonalika", 0.9), ("swaraj", 0.9),
        ("eicher", 0.9), ("farmtrac", 0.9), ("john deere", 0.9),
        ("clutch", 0.7), ("hydraulic", 0.7), ("pto", 0.6),
        ("bonnet", 0.6), ("gear", 0.5), ("plough", 0.5),
    ],
    "harvester": [
        ("harvester", 1.0), ("combine", 1.0), ("हार्वेस्टर", 1.0),
        ("claas", 0.9), ("reaper", 0.9), ("header", 0.8),
        ("cutter bar", 0.8), ("grain tank", 0.8), ("feeder house", 0.7),
        ("anaj katai", 0.8), ("combine harvester", 1.0),
    ],
    "thresher": [
        ("thresher", 1.0), ("थ्रेशर", 1.0), ("threshing machine", 1.0),
        ("gahai", 0.9), ("गाहना", 0.9), ("bhusa", 0.8), ("भूसा", 0.8),
        ("drum jam", 0.8), ("cylinder jam", 0.8),
        ("concave", 0.7), ("sieve", 0.6), ("chalni", 0.7),
    ],
    "submersible_pump": [
        ("submersible", 1.0), ("borewell", 1.0), ("bore pump", 1.0),
        ("tubewell", 0.9), ("patal pump", 0.9), ("सबमर्सिबल", 1.0),
        ("borewell motor", 0.9), ("underground pump", 0.8),
        ("three phase", 0.5), ("mcb trip", 0.7), ("starter panel", 0.7),
    ],
    "water_pump": [
        ("water pump", 1.0), ("pump set", 0.9), ("monoblock", 0.9),
        ("centrifugal pump", 0.9), ("pani pump", 0.9),
        ("surface pump", 0.8), ("suction pipe", 0.8),
        ("foot valve", 0.8), ("priming", 0.7), ("irrigation pump", 0.8),
    ],
    "electric_motor": [
        ("electric motor", 1.0), ("bijli motor", 0.9),
        ("motor", 0.3),            # was 0.5 — farmers say "motor" for any machine;
                                   # reduced so "borewell motor" / "water pump motor"
                                   # still classify correctly via their own keywords
        ("induction motor", 0.9), ("three phase motor", 0.9),
        ("capacitor", 0.8), ("winding", 0.7), ("overload relay", 0.8),
        ("single phase motor", 0.8), ("terminal box", 0.7),
    ],
    "power_tiller": [
        ("power tiller", 1.0), ("walking tractor", 0.9),
        ("mini tractor", 0.8), ("vst shakti", 0.9),
        ("chhota tractor", 0.8), ("hand tractor", 0.8),
        ("tiller", 0.6), ("puddling", 0.7), ("decompression", 0.6),
    ],
    "rotavator": [
        ("rotavator", 1.0), ("rotary tiller", 0.9), ("rotovator", 0.9),
        ("rotary cultivator", 0.9), ("rota", 0.7),
        ("l blade", 0.8), ("shear bolt", 0.9),
        ("blade tod", 0.7), ("blade toot gayi", 0.8),
    ],
    "chaff_cutter": [
        ("chaff cutter", 1.0), ("toka machine", 1.0), ("toka", 0.9),
        ("चारा काटने", 1.0), ("fodder cutter", 0.9), ("hay cutter", 0.8),
        ("bhusa machine", 0.9), ("chara kaat", 0.9), ("blade dull", 0.7),
    ],
    "generator": [
        ("generator", 1.0), ("genset", 1.0), ("जनरेटर", 1.0),
        ("light plant", 0.9), ("bijli generator", 0.9),
        ("alternator", 0.8), ("avr", 0.8),
        ("current nahi", 0.7), ("light nahi", 0.6), ("voltage drop", 0.6),
    ],
    "diesel_engine": [
        ("diesel engine", 1.0), ("stationary engine", 0.9),
        ("डीजल इंजन", 1.0), ("lister engine", 0.9),
        ("kirloskar", 0.9), ("flywheel", 0.8),
        ("decompression lever", 0.8), ("hand crank", 0.7),
        ("standalone engine", 0.8), ("lombardini", 0.8),
    ],
    # ── New machines ─────────────────────────────────────────────────────────
    "cultivator": [
        ("cultivator", 1.0), ("कल्टीवेटर", 1.0),
        ("tine", 0.9), ("soil tiller", 0.9), ("weeder", 0.8),
        ("inter cultivation", 0.9), ("kulti", 0.9), ("kultivator", 0.9),
        ("dantali", 0.8), ("दंताली", 0.8),          # common Hindi/Gujarati term
        ("shovel tine", 0.8), ("sweep", 0.7),
    ],
    "sprayer": [
        ("sprayer", 1.0), ("स्प्रेयर", 1.0), ("spray machine", 1.0),
        ("pesticide machine", 1.0), ("keetnaashak machine", 0.9),
        ("knapsack", 0.9), ("boom sprayer", 1.0), ("power sprayer", 0.9),
        ("nozzle", 0.7), ("pump spray", 0.8), ("dawa machine", 0.9),
        ("दवाई मशीन", 0.9), ("छिड़काव", 0.9),         # Hindi: "chidkav" = spraying
        ("fog machine", 0.8), ("mist blower", 0.8),
    ],
    "drip_irrigation": [
        ("drip irrigation", 1.0), ("drip system", 1.0), ("ड्रिप", 1.0),
        ("micro irrigation", 1.0), ("sprinkler", 0.9), ("स्प्रिंकलर", 0.9),
        ("dripper", 0.9), ("lateral pipe", 0.8), ("emitter", 0.8),
        ("filter unit", 0.8), ("fertigation", 0.8), ("pressure regulator", 0.7),
        ("paani pipe", 0.6), ("seench pipe", 0.7), ("सिंचाई पाइप", 0.8),
        ("rain gun", 0.9), ("mini sprinkler", 0.9),
    ],
}

# Flat list for O(n) scan: (keyword, machine_id, weight)
_KEYWORD_INDEX: list[tuple[str, str, float]] = [
    (kw.lower(), mid, wt)
    for mid, entries in _AUDIO_KEYWORDS.items()
    for kw, wt in entries
]

# Max possible score per machine (sum of all weights)
_MAX_AUDIO_SCORE: dict[str, float] = {
    mid: sum(wt for _, wt in entries)
    for mid, entries in _AUDIO_KEYWORDS.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _ClsResult:
    machine_type: str
    confidence: float
    source: str
    all_scores: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Returned to main.py — contains the final resolved machine_type."""
    machine_type:     str
    confidence:       float
    source:           str    # "clip" | "audio" | "fused" | "gemini" | "hint" | "default"
    clip_confidence:  float
    audio_confidence: float
    gemini_used:      bool
    frame_used:       Optional[str]        # "early" | "mid" | None
    frames:           list = field(default_factory=list)
    # ↑ JPEG bytes of the already-decoded PIL frames (early/mid/late order).
    # Populated by detect_machine() at zero extra cost — the PIL images were
    # already in memory from _extract_frames(). main.py reads these directly
    # for Gemini diagnosis instead of re-opening and re-decoding the video.


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL LOADING  (called once from lifespan)
# ══════════════════════════════════════════════════════════════════════════════

def load_clip_model() -> bool:
    """
    Load MobileCLIP-S1 via open_clip and pre-compute text embeddings for 11 machines.
    Must be called once from FastAPI lifespan() before any requests arrive.

    open_clip API differences from HF transformers:
      - create_model_and_transforms() returns (model, train_transform, eval_transform)
      - Images go through eval_transform (torchvision pipeline) → tensor directly
      - Text is tokenized with open_clip.get_tokenizer(model_name)
      - Embeddings: model.encode_image(tensor) / model.encode_text(tokens)
      - No CLIPProcessor — the preprocess transform replaces it entirely

    Performance (HuggingFace free tier, 2-core CPU):
      Model download: ~50 MB (first run, then cached)
      Load time:      ~1–2 seconds
      Per-frame inference: ~22 ms
      3 frames: ~66 ms total  (vs ~210 ms for 3 × ViT-B/32)

    Multi-worker note: workers=1 in uvicorn → safe. If you scale to workers>1,
    each worker loads its own ~50 MB copy (~200 MB each including torch) — fine.
    """
    global _clip_model, _clip_preprocess, _clip_tokenizer, _text_embeddings, _clip_ready

    t0 = time.time()
    try:
        import torch
        import torch.nn.functional as F
        import open_clip

        logger.info(f"🔮 Loading MobileCLIP: {_MODEL_NAME} pretrained={_MODEL_PRETRAINED} device={_DEVICE}")

        # Validate the pretrained tag exists BEFORE the 50 MB download starts.
        # open_clip.list_pretrained() returns [(model_name, tag), ...].
        # A bad env var (e.g. CLIP_PRETRAINED=datacomp_dr instead of datacompdr)
        # would otherwise cause a cryptic RuntimeError deep inside open_clip.
        available = open_clip.list_pretrained()
        valid_tags = [tag for name, tag in available if name == _MODEL_NAME]
        if not valid_tags:
            raise ValueError(
                f"Model '{_MODEL_NAME}' not found in open_clip registry. "
                f"Available MobileCLIP models: "
                f"{[n for n, _ in available if 'MobileCLIP' in n]}"
            )
        if _MODEL_PRETRAINED not in valid_tags:
            raise ValueError(
                f"Pretrained tag '{_MODEL_PRETRAINED}' not valid for '{_MODEL_NAME}'. "
                f"Valid tags: {valid_tags}  "
                f"(Check CLIP_PRETRAINED env var — correct value is 'datacompdr')"
            )
        logger.info(f"✅ Pretrained tag '{_MODEL_PRETRAINED}' verified for '{_MODEL_NAME}'")

        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME,
            pretrained=_MODEL_PRETRAINED,
        )
        _clip_model = _clip_model.to(_DEVICE)
        _clip_model.eval()

        _clip_tokenizer = open_clip.get_tokenizer(_MODEL_NAME)

        # Pre-compute and cache text embeddings — once at startup, ~0.1s
        logger.info("📝 Pre-computing text embeddings for 11 machines...")
        _text_embeddings = {}
        with torch.no_grad():
            for machine_id, prompts in _MACHINE_PROMPTS.items():
                tokens = _clip_tokenizer(prompts).to(_DEVICE)          # [n_prompts, ctx]
                feats  = _clip_model.encode_text(tokens)                # [n_prompts, D]
                avg    = feats.mean(dim=0)                              # [D]
                _text_embeddings[machine_id] = F.normalize(avg, dim=-1)

        _clip_ready = True
        elapsed = time.time() - t0
        logger.info(
            f"✅ MobileCLIP-S1 ready in {elapsed:.1f}s — "
            f"{len(_text_embeddings)} machine embeddings cached  "
            f"[model={_MODEL_NAME} pretrained={_MODEL_PRETRAINED}]"
        )
        return True

    except ImportError as exc:
        logger.warning(
            f"⚠️  MobileCLIP not available ({exc}). "
            "Detection falls back to audio keywords + Gemini. "
            "Fix: pip install open_clip_torch"
        )
        return False
    except Exception as exc:
        logger.error(f"❌ MobileCLIP load failed: {exc}")
        return False

@dataclass
class _Frame:
    image:          object   # PIL.Image.Image (center-cropped, max 448px)
    label:          str      # "early" | "mid" | "late"
    blur_score:     float    # Laplacian variance — higher = sharper
    brightness:     float    # mean pixel value 0–255
    usable:         bool


def _center_crop_70(bgr) -> object:
    """
    Crop the center 70% of a frame.
    Removes sky/ground/border clutter that confuses CLIP.
    Returns cropped BGR ndarray.
    """
    import cv2 as _cv2
    h, w = bgr.shape[:2]
    # 70% crop: remove 15% from each side
    margin_y = int(h * 0.15)
    margin_x = int(w * 0.15)
    return bgr[margin_y:h - margin_y, margin_x:w - margin_x]


def _extract_frames(video_path: Path) -> list[_Frame]:
    """
    Extract 3 frames from video using OpenCV.
      - Frame 1: 15% into video (establishing wide shot)
      - Frame 2: 40% into video (typically machine close-up begins)
      - Frame 3: 70% into video (late-clip confirmation angle)

    Per frame:
      1. Center-crop to 70% — removes background edges
      2. Blur detection (Laplacian variance)
      3. Brightness check + CLAHE if too dark
      4. Resize to max 448px (optimal for ViT-L/14 vs 512px for ViT-B/32)

    Returns list of _Frame objects (may be empty if video unreadable).
    """
    import cv2
    from PIL import Image as PILImage

    # MobileCLIP-S1 native input resolution is 256 × 256 px.
    # Feeding 256px directly (vs 448px for ViT-L/14) is faster and correct.
    # The open_clip preprocess transform will resize to 256px anyway.
    _MAX_PX = 256

    frames: list[_Frame] = []
    cap = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"⚠️  Cannot open video: {video_path.name}")
            return frames

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 3:
            logger.warning(f"⚠️  Video too short ({total} frames)")
            return frames

        for idx_pct, label in [(0.15, "early"), (0.40, "mid"), (0.70, "late")]:
            frame_idx = max(0, int(total * idx_pct))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                logger.warning(f"⚠️  Could not read frame at {idx_pct*100:.0f}%")
                continue

            # Center-crop to 70% BEFORE quality checks — so blur/brightness
            # are measured on the cropped region (the part CLIP will actually see)
            bgr = _center_crop_70(bgr)

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            blur       = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())

            # CLAHE if too dark
            if brightness < _BRIGHT_MIN:
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.createCLAHE(
                    clipLimit=2.0, tileGridSize=(8, 8)
                ).apply(lab[:, :, 0])
                bgr        = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                gray       = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                logger.info(f"🔆 Frame [{label}] CLAHE → brightness {brightness:.1f}")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            pil.thumbnail((_MAX_PX, _MAX_PX), PILImage.Resampling.LANCZOS)

            usable = blur >= _BLUR_MIN and brightness >= _BRIGHT_MIN
            logger.info(
                f"📸 Frame [{label}] blur={blur:.1f} bright={brightness:.1f} "
                f"size={pil.size} usable={usable}"
            )
            frames.append(_Frame(pil, label, blur, brightness, usable))

    except Exception as exc:
        logger.error(f"❌ Frame extraction: {exc}")
    finally:
        if cap is not None:
            cap.release()

    return frames


def _best_frames(frames: list[_Frame], n: int = 2) -> list[_Frame]:
    """
    Return the top-n sharpest usable frames for multi-frame averaging.
    Falls back to top-n by blur score if not enough usable frames.
    """
    if not frames:
        return []
    usable = [f for f in frames if f.usable]
    pool   = usable if usable else frames
    sorted_frames = sorted(pool, key=lambda f: f.blur_score, reverse=True)
    return sorted_frames[:n]


# Keep the old name as alias for detect_machine compatibility
def _best_frame(frames: list[_Frame]) -> Optional[_Frame]:
    """Legacy single-frame selector — used by detect_machine for Gemini fallback."""
    result = _best_frames(frames, n=1)
    return result[0] if result else None


def _frames_to_jpeg(frames: list[_Frame], max_px: int = 512) -> list[bytes]:
    """
    JPEG-encode the PIL images that _extract_frames() already decoded.

    This is the zero-cost path: the frames are already in RAM as PIL.Image
    objects. We just resize to 512px (for Gemini — larger than CLIP's 256px
    to preserve visual detail) and encode to JPEG bytes.

    No video file is re-opened. No cv2 decode. Pure in-memory PIL operations.
    Cost: ~2–5 ms per frame for the JPEG encode step.

    Preserves original label order: [early, mid, late].
    """
    import io as _io
    result: list[bytes] = []
    for fr in frames:
        try:
            img = fr.image.copy()
            img.thumbnail((max_px, max_px))
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            result.append(buf.getvalue())
        except Exception as exc:
            logger.warning(f"⚠️  Frame [{fr.label}] JPEG encode failed: {exc}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CLIP IMAGE CLASSIFICATION  (multi-frame averaged)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_image(frames: list[_Frame]) -> _ClsResult:
    """
    Run MobileCLIP zero-shot classification on 1–3 frames, average embeddings.

    open_clip image encoding:
      _clip_preprocess(pil_image) → torch.Tensor [3, H, W]
      _clip_model.encode_image(tensor.unsqueeze(0)) → [1, D]

    Multi-frame averaging: stack N image embeddings → mean → L2-normalise →
    dot with pre-cached text embeddings → softmax(T=50) → probabilities.

    Called synchronously — wrapped in run_in_executor by detect_machine().
    """
    if not _clip_ready or not frames:
        return _ClsResult("tractor", 0.0, "clip_unavailable")

    try:
        import torch
        import torch.nn.functional as F

        t0 = time.time()
        img_feats = []

        with torch.no_grad():
            for fr in frames:
                # open_clip preprocess: PIL.Image → normalised tensor [3, 256, 256]
                tensor = _clip_preprocess(fr.image).unsqueeze(0).to(_DEVICE)  # [1, 3, H, W]
                feat   = _clip_model.encode_image(tensor)                      # [1, D]
                # Normalize before averaging — defensive: open_clip encode_image
                # outputs unit vectors by default, but normalizing here ensures
                # correctness if model or preprocess is swapped via env var.
                # Cost: one F.normalize per frame (~microseconds). Zero downside.
                feat   = F.normalize(feat, dim=-1)                             # [1, D] unit vector
                img_feats.append(feat)

            # Spherical mean: average of unit vectors, then re-normalize.
            # Mathematically equivalent to avg→norm when inputs are already unit
            # vectors, but this order is the geometrically correct definition of
            # a mean direction on the unit hypersphere.
            avg_feat = torch.stack(img_feats, dim=0).mean(dim=0)   # [1, D]
            avg_feat = F.normalize(avg_feat, dim=-1)                # [1, D] unit vector

            machine_ids = list(_text_embeddings.keys())
            txt_stack   = torch.stack(
                [_text_embeddings[m] for m in machine_ids]
            )                                                        # [11, D]
            sims  = (avg_feat @ txt_stack.T).squeeze(0)             # [11] cosine sims

            # Temperature calibration — MobileCLIP-S1 specific.
            # Calibrated from production data (2026-03-12):
            #   Actual raw cosine sim gap (tractor vs power_tiller) = 0.026
            #   At T=30: softmax top = 0.372 — below _CLIP_STRONG=0.36, so the
            #             second detection pass fires and audio-only (0.155) would
            #             overwrite the correct CLIP result.
            #   At T=50: softmax top = 0.649 — above threshold, second pass
            #             skipped, tractor confirmed in a single pass. ✅
            #
            # Routing behaviour at T=50 across gap sizes:
            #   Strong match  (gap ~0.060): top ~0.92 → clip_dominant ✅
            #   Clear match   (gap ~0.026): top ~0.65 → clip_dominant ✅
            #   Ambiguous     (gap ~0.005): top ~0.20 → fused_weak → Gemini ✅
            #
            # T=50 works for all MobileCLIP variants (S1, S2, B) because
            # the raw sim distributions are consistent across model sizes.
            probs = torch.softmax(sims * 50.0, dim=0).cpu().numpy()

        all_scores  = {mid: float(p) for mid, p in zip(machine_ids, probs)}
        top_idx     = int(probs.argmax())
        top_machine = machine_ids[top_idx]
        top_conf    = float(probs[top_idx])
        second      = machine_ids[int(probs.argsort()[-2])]

        logger.info(
            f"🔮 MobileCLIP ({len(frames)}fr avg): {top_machine}={top_conf:.3f}  "
            f"#2={second}={float(sorted(probs)[-2]):.3f}  "
            f"[{(time.time()-t0)*1000:.0f}ms]"
        )
        return _ClsResult(top_machine, top_conf, "clip", all_scores)

    except Exception as exc:
        logger.error(f"❌ MobileCLIP inference: {exc}")
        return _ClsResult("tractor", 0.0, "clip_error")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  AUDIO KEYWORD CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def _classify_audio(text: str) -> _ClsResult:
    """
    Score each machine type by keywords found in transcription text.
    Zero API calls. Pure string match. Runtime < 1ms.

    Confidence = matched_weight_sum / max_possible_weight  (capped at 1.0)
    Multiple keyword hits increase confidence proportionally.
    """
    if not text:
        return _ClsResult("tractor", 0.0, "audio_empty")

    lower = text.lower()
    raw: dict[str, float] = {mid: 0.0 for mid in _AUDIO_KEYWORDS}

    for kw, mid, wt in _KEYWORD_INDEX:
        if kw in lower:
            raw[mid] += wt

    if not any(raw.values()):
        return _ClsResult("tractor", 0.0, "audio_no_match")

    norm = {mid: min(raw[mid] / _MAX_AUDIO_SCORE[mid], 1.0) for mid in raw}
    top  = max(norm, key=norm.get)
    logger.info(f"🎙️  Audio: {top}={norm[top]:.3f}")
    return _ClsResult(top, norm[top], "audio", norm)

@dataclass
class _FusionResult:
    machine_type:  str
    confidence:    float
    source:        str
    needs_gemini:  bool


def _fuse(clip: Optional[_ClsResult], audio: Optional[_ClsResult]) -> _FusionResult:
    ct  = clip.machine_type  if clip  else None
    cc  = clip.confidence    if clip  else 0.0
    at  = audio.machine_type if audio else None
    ac  = audio.confidence   if audio else 0.0
    agree = ct is not None and at is not None and ct == at

    # Both strong and agree
    if cc >= _CLIP_STRONG and ac >= _AUDIO_WEAK and agree:
        fused = min(cc * 0.7 + ac * 0.3, 1.0)
        logger.info(f"✅ Fused AGREE {ct}  conf={fused:.3f}")
        return _FusionResult(ct, fused, "fused_agree", False)

    # Both strong but disagree — can't trust either
    if cc >= _CLIP_STRONG and ac >= _AUDIO_STRONG and not agree:
        logger.warning(f"⚠️  Fused CONTRADICT  clip={ct}({cc:.2f}) audio={at}({ac:.2f})")
        return _FusionResult(ct, cc, "fused_contradict", True)

    # CLIP strong alone
    if cc >= _CLIP_STRONG:
        logger.info(f"✅ CLIP dominant {ct}={cc:.3f}")
        return _FusionResult(ct, cc, "clip_dominant", False)

    # Audio strong alone
    if ac >= _AUDIO_STRONG and at:
        logger.info(f"✅ Audio dominant {at}={ac:.3f}")
        return _FusionResult(at, ac, "audio_dominant", False)

    # Both moderate and agree
    if cc >= _CLIP_WEAK and ac >= _AUDIO_WEAK and agree:
        avg = (cc + ac) / 2.0
        needs = avg < _FUSED_MIN
        logger.info(f"{'✅' if not needs else '⚠️ '} Fused MODERATE {ct}={avg:.3f} gemini={needs}")
        return _FusionResult(ct, avg, "fused_moderate", needs)

    # Both weak or disagree
    best_t = ct if cc >= ac else at
    best_c = max(cc, ac)
    logger.info(f"⚠️  Fused WEAK — gemini needed  best_guess={best_t}({best_c:.2f})")
    return _FusionResult(best_t or "tractor", best_c, "fused_weak", True)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  GEMINI VISION FALLBACK
# ══════════════════════════════════════════════════════════════════════════════
#
# Called ONLY when fusion.needs_gemini is True AND a video frame is available.
# Prompt is intentionally minimal: ~80 tokens total.
# Returns machine_type string matching one of the 11 canonical IDs.

_ALL_MACHINE_IDS = list(_MACHINE_PROMPTS.keys())

async def _gemini_fallback(frame: _Frame, prior: Optional[str] = None) -> _ClsResult:
    """
    Minimal Gemini Vision call: 1 image + 1-line prompt ≈ 85 tokens.

    Prompt requests JSON so the response is unambiguous — no sentence answers,
    no "I cannot determine", no partial labels. Falls back to substring match
    if JSON parsing fails (defensive, should never happen in practice).

    Multi-worker note: if you ever scale to workers>1, each worker holds its
    own CLIP model instance in memory (~150 MB each). This is fine for
    HuggingFace Spaces (workers=1). Document this before scaling.
    """
    import json as _json
    try:
        import google.generativeai as genai

        labels = " | ".join(_ALL_MACHINE_IDS)
        hint   = f" It is likely a {prior}." if prior else ""
        # JSON prompt eliminates sentence answers, refusals, partial labels
        prompt = (
            f'Which farm machine is in this image?{hint} '
            f'Return ONLY this JSON: {{"machine_type": "<label>"}} '
            f'where <label> is exactly one of: {labels}'
        )
        model    = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate_content([prompt, frame.image]),
        )
        raw_text = response.text.strip()

        # Primary parser: JSON
        matched = None
        try:
            # Strip markdown fences if Gemini adds them despite instruction
            clean = raw_text.replace("```json", "").replace("```", "").strip()
            data  = _json.loads(clean)
            label = str(data.get("machine_type", "")).lower().strip().replace(" ", "_").replace("-", "_")
            matched = next((m for m in _ALL_MACHINE_IDS if m == label), None)
        except (_json.JSONDecodeError, AttributeError):
            pass

        # Fallback parser: substring match (handles rare non-JSON responses)
        if not matched:
            normalised = raw_text.lower().replace(" ", "_").replace("-", "_")
            matched = next(
                (m for m in _ALL_MACHINE_IDS if m in normalised or normalised in m),
                None,
            )

        if matched:
            logger.info(f"🔮 Gemini fallback: {matched}")
            return _ClsResult(matched, 0.82, "gemini", {matched: 0.82})

        logger.warning(f"⚠️  Gemini returned unmatched response: '{raw_text[:60]}'")
        return _ClsResult(prior or "tractor", 0.40, "gemini_unmatched")

    except Exception as exc:
        logger.error(f"❌ Gemini fallback: {exc}")
        return _ClsResult(prior or "tractor", 0.30, "gemini_error")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  PUBLIC API — called from main.py
# ══════════════════════════════════════════════════════════════════════════════

async def detect_machine(
    video_path: Optional[Path],
    transcription_text: str,
) -> DetectionResult:
    """
    Full detection pipeline. Call this after audio transcription, before diagnosis.

    Args:
        video_path:         saved video file (or None if no video provided)
        transcription_text: output of transcribe_audio_with_gemini()

    Returns:
        DetectionResult — machine_type is always a valid registry canonical ID.

    Token cost:
        Normal (CLIP+audio agree or one dominant):  0 tokens
        Fallback (Gemini needed):                  ~80 tokens
    """
    t0 = time.time()

    # Step 1 — Audio (free, always run first)
    audio_result = _classify_audio(transcription_text)

    # Step 2 — Frame extraction (3 frames: 15% / 40% / 70%)
    all_frames: list[_Frame] = []
    top_frames: list[_Frame] = []   # best 2 for CLIP averaging
    best_single: Optional[_Frame] = None  # best 1 for Gemini fallback

    if video_path and video_path.exists() and video_path.stat().st_size > 1024:
        all_frames = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _extract_frames(video_path)
        )
        top_frames   = _best_frames(all_frames, n=2)
        best_single  = _best_frame(all_frames)   # for Gemini fallback only
    else:
        logger.info("📹 No usable video — image classification skipped")

    # Step 3 — CLIP on averaged top-2 frames (or 1 if only 1 extracted)
    clip_result: Optional[_ClsResult] = None
    if top_frames and _clip_ready:
        clip_result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _classify_image(top_frames)
        )

    # Step 4 — Fusion
    fusion = _fuse(clip_result, audio_result)

    # Step 5 — Gemini fallback (only when needed AND frame available)
    gemini_used = False
    final       = fusion

    if fusion.needs_gemini:
        if best_single is not None:
            gem = await _gemini_fallback(
                best_single,
                prior=fusion.machine_type if fusion.confidence > 0.20 else None,
            )
            final = _FusionResult(gem.machine_type, gem.confidence, "gemini", False)
            gemini_used = True
        elif audio_result.confidence >= _AUDIO_WEAK:
            # No video at all — trust audio
            final = _FusionResult(
                audio_result.machine_type, audio_result.confidence, "audio_no_video", False
            )
            logger.info(f"🎙️  No video — audio only: {audio_result.machine_type}")
        # else: keep fusion's best guess

    # Step 6 — Validate against registry
    from utils.machine_registry import get_profile, resolve_machine_id
    canonical = resolve_machine_id(final.machine_type)
    if not get_profile(canonical):
        logger.warning(f"⚠️  '{canonical}' not in registry — defaulting to tractor")
        canonical = "tractor"

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        f"✅ detect_machine: {canonical}  conf={final.confidence:.3f}  "
        f"source={final.source}  gemini={gemini_used}  [{elapsed_ms:.0f}ms]"
    )

    # JPEG-encode the frames that are already in RAM — zero extra video decode.
    # main.py reads detection.frames directly; _extract_clip_frames() is gone.
    jpeg_frames = _frames_to_jpeg(all_frames) if all_frames else []

    return DetectionResult(
        machine_type     = canonical,
        confidence       = round(final.confidence, 4),
        source           = final.source,
        clip_confidence  = round(clip_result.confidence  if clip_result  else 0.0, 4),
        audio_confidence = round(audio_result.confidence if audio_result else 0.0, 4),
        gemini_used      = gemini_used,
        frame_used       = "+".join(f.label for f in top_frames) if top_frames else None,
        frames           = jpeg_frames,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 12.  FAST PATH — pre-extracted frames from Flutter client
# ══════════════════════════════════════════════════════════════════════════════
#
# Flutter extracts 3 frames locally (at 15% / 40% / 70% of video duration)
# and uploads them as three ~50 KB JPEGs instead of a 20–35 MB video file.
# This function accepts those JPEG bytes and runs the IDENTICAL CLIP → fusion
# → Gemini pipeline as detect_machine() — the only difference is how _Frame
# objects are constructed (from JPEG bytes instead of cv2.VideoCapture).
#
# Accuracy: identical — CLIP receives the same 256px PIL images.
# Token usage: identical — same fusion logic, same Gemini fallback threshold.
# The returned DetectionResult.frames list is the SAME JPEG bytes passed in,
# re-encoded at 512px for Gemini (matching what _frames_to_jpeg() produces).

async def detect_machine_from_frames(
    frame_bytes_list: list[bytes],   # 1–3 JPEG bytes, [early, mid, late] order
    transcription_text: str,
) -> DetectionResult:
    """
    Detection pipeline for pre-extracted frames (fast-path from Flutter).

    Identical to detect_machine() in every way except frame sourcing:
      - No video file saved to disk
      - No cv2.VideoCapture decode
      - No _extract_frames() call
      Replaces those with direct PIL.Image.open() on the uploaded JPEG bytes.

    The resulting _Frame objects feed into the EXACT SAME:
      _best_frames() → _classify_image() → _fuse() → _gemini_fallback() chain.

    Args:
        frame_bytes_list:   1–3 JPEG byte strings, [early, mid, late] order
        transcription_text: output of transcribe_audio_with_gemini()

    Returns:
        DetectionResult — identical structure to detect_machine()
    """
    import io as _io
    from PIL import Image as _PILImage

    t0 = time.time()
    _LABELS = ["early", "mid", "late"]

    # Step 1 — Audio classification (identical to detect_machine)
    audio_result = _classify_audio(transcription_text)

    # Step 2 — Reconstruct _Frame objects from the JPEG bytes Flutter uploaded.
    # We perform the same quality checks (blur, brightness) so the frame
    # selection logic (_best_frames, _best_frame) behaves identically.
    all_frames: list[_Frame] = []
    for i, jpeg_bytes in enumerate(frame_bytes_list[:3]):
        label = _LABELS[i] if i < len(_LABELS) else f"frame{i}"
        try:
            import cv2 as _cv2
            import numpy as _np

            pil = _PILImage.open(_io.BytesIO(jpeg_bytes)).convert("RGB")
            # Resize to 256px — identical to _extract_frames() _MAX_PX target
            pil.thumbnail((256, 256), _PILImage.Resampling.LANCZOS)

            # Blur + brightness on the same pixel data CLIP will see
            bgr = _cv2.cvtColor(_np.array(pil), _cv2.COLOR_RGB2BGR)
            gray = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
            blur       = float(_cv2.Laplacian(gray, _cv2.CV_64F).var())
            brightness = float(gray.mean())

            # CLAHE if too dark — identical to _extract_frames()
            if brightness < _BRIGHT_MIN:
                lab = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = _cv2.createCLAHE(
                    clipLimit=2.0, tileGridSize=(8, 8)
                ).apply(lab[:, :, 0])
                bgr        = _cv2.cvtColor(lab, _cv2.COLOR_LAB2BGR)
                gray       = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                pil        = _PILImage.fromarray(
                    _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                )
                logger.info(f"🔆 Pre-frame [{label}] CLAHE → brightness {brightness:.1f}")

            usable = blur >= _BLUR_MIN and brightness >= _BRIGHT_MIN
            logger.info(
                f"📸 Pre-frame [{label}] blur={blur:.1f} bright={brightness:.1f} "
                f"size={pil.size} usable={usable}"
            )
            all_frames.append(_Frame(pil, label, blur, brightness, usable))

        except Exception as exc:
            logger.warning(f"⚠️  Pre-frame [{label}] decode failed: {exc}")

    if not all_frames:
        logger.warning("⚠️  All pre-extracted frames failed decode — falling back to audio only")

    # Steps 3–6: IDENTICAL to detect_machine() from here onward ───────────────

    top_frames  = _best_frames(all_frames, n=2)
    best_single = _best_frame(all_frames)

    clip_result: Optional[_ClsResult] = None
    if top_frames and _clip_ready:
        clip_result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _classify_image(top_frames)
        )

    fusion = _fuse(clip_result, audio_result)

    gemini_used = False
    final       = fusion

    if fusion.needs_gemini:
        if best_single is not None:
            gem = await _gemini_fallback(
                best_single,
                prior=fusion.machine_type if fusion.confidence > 0.20 else None,
            )
            final = _FusionResult(gem.machine_type, gem.confidence, "gemini", False)
            gemini_used = True
        elif audio_result.confidence >= _AUDIO_WEAK:
            final = _FusionResult(
                audio_result.machine_type, audio_result.confidence, "audio_no_video", False
            )
            logger.info(f"🎙️  No usable frames — audio only: {audio_result.machine_type}")

    from utils.machine_registry import get_profile, resolve_machine_id
    canonical = resolve_machine_id(final.machine_type)
    if not get_profile(canonical):
        logger.warning(f"⚠️  '{canonical}' not in registry — defaulting to tractor")
        canonical = "tractor"

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        f"✅ detect_machine_from_frames: {canonical}  conf={final.confidence:.3f}  "
        f"source={final.source}  gemini={gemini_used}  [{elapsed_ms:.0f}ms]"
    )

    # Re-encode at 512px for Gemini — identical to _frames_to_jpeg()
    jpeg_frames = _frames_to_jpeg(all_frames) if all_frames else list(frame_bytes_list[:3])

    return DetectionResult(
        machine_type     = canonical,
        confidence       = round(final.confidence, 4),
        source           = final.source,
        clip_confidence  = round(clip_result.confidence  if clip_result  else 0.0, 4),
        audio_confidence = round(audio_result.confidence if audio_result else 0.0, 4),
        gemini_used      = gemini_used,
        frame_used       = "+".join(f.label for f in top_frames) if top_frames else None,
        frames           = jpeg_frames,
    )