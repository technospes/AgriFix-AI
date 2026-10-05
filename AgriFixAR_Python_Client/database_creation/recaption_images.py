"""Re-run image captioning only — skips text extraction."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_manuals import (
    extract_images_from_pdf,
    caption_images_with_gemini,
    parse_pdf_to_markdown,
    _save_image_knowledge_db,
    _gemini_throttle,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PDF_FOLDER = Path(r"D:\AgriFix_Workspace\AgriFixAR_Python_Client\database_creation\power_tiller_Pdfs")
IMAGE_DB = PDF_FOLDER / "Master_Power_Tiller_DB_images.json"

# Load existing images (if any)
existing_images = []
if IMAGE_DB.exists():
    with open(IMAGE_DB, 'r', encoding='utf-8') as f:
        existing_images = json.load(f)
    logger.info(f"Loaded {len(existing_images)} existing image records")

all_images = []
for pdf_path in PDF_FOLDER.glob("*.pdf"):
    logger.info(f"Processing images from: {pdf_path.name}")
    
    # Build page→section map from markdown
    md_text = parse_pdf_to_markdown(str(pdf_path))
    page_section_map = {}
    current_page, current_section = 1, "General"
    for line in md_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            current_section = stripped.lstrip("#").strip() or current_section
        if current_page not in page_section_map:
            page_section_map[current_page] = current_section
    
    # Extract and caption
    img_dir = PDF_FOLDER / "extracted_images"
    raw_images = extract_images_from_pdf(pdf_path, img_dir, page_section_map)
    
    if raw_images:
        logger.info(f"   {len(raw_images)} candidates → captioning with Gemini Flash Lite...")
        captioned = caption_images_with_gemini(raw_images, img_dir)
        succeeded = sum(1 for c in captioned if c.get("caption"))
        logger.info(f"   {succeeded}/{len(captioned)} captioned successfully")
        all_images.extend(captioned)
    else:
        logger.info(f"   No images survived filtering")

# Merge with existing and save
all_images = existing_images + all_images
_save_image_knowledge_db(all_images, IMAGE_DB)
logger.info(f"✅ Image DB saved: {len(all_images)} total records → {IMAGE_DB}")