import fitz
import os
import hashlib
from PIL import Image
import io
import json
from pathlib import Path
import logging
from datetime import datetime
import sys
import base64
import google.generativeai as genai
from typing import Dict, List, Tuple
import time

# =============== CONFIGURATION ===============
class Config:
    # PDF Settings
    PDF_PATH = "knowledge_base/Mahindra/Tractors_part_all images.pdf"
    OUTPUT_ROOT = "extracted_images_ai_categorized"
    
    # Gemini API - Set your API key here OR use environment variable GEMINI_API_KEY
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Reads from environment variable
    GEMINI_MODEL = "gemini-2.0-flash"  # or "gemini-1.5-pro" for better accuracy
    
    # Image Filters
    MIN_IMAGE_SIZE_KB = 10
    MIN_DIMENSION = 100
    MAX_IMAGES_PER_PAGE = 20
    SUPPORTED_FORMATS = {'jpeg', 'jpg', 'png', 'gif', 'bmp', 'tiff'}
    
    # Quality Settings
    COMPRESSION_QUALITY = 85
    RESIZE_LARGE_IMAGES = True
    MAX_IMAGE_DIMENSION = 2048
    
    # AI Settings
    AI_RETRY_ATTEMPTS = 3
    AI_RETRY_DELAY = 10  # seconds between retries
    BATCH_SIZE = 5  # Process images in batches to avoid rate limits
    RATE_LIMIT_DELAY = 7  # seconds between API calls (free tier: 10 req/min = 6 sec minimum)
    
    # Categories for organization
    EXPECTED_CATEGORIES = [
        'battery', 'engine', 'fuel_tank', 'transmission', 'hydraulics',
        'electrical', 'cooling_system', 'exhaust', 'filters', 'wheels_tires',
        'steering', 'brakes', 'lights', 'controls', 'seats', 'cabin',
        'pto', 'hitch', 'frame_chassis', 'diagram', 'schematic', 'unknown'
    ]
    
    # Logging
    LOG_FILE = "ai_extraction.log"
    METADATA_FILE = "ai_extraction_metadata.json"

# =============== SETUP ===============
def setup_logging():
    """Configure logging with Windows encoding support"""
    if sys.platform.startswith('win'):
        import locale
        if locale.getpreferredencoding().lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def setup_gemini():
    """Initialize Gemini API"""
    api_key = Config.GEMINI_API_KEY
    
    # Check if API key is set
    if not api_key or api_key.strip() == "":
        raise ValueError(
            "\n" + "="*70 + "\n"
            "❌ GEMINI API KEY NOT FOUND!\n"
            "="*70 + "\n"
            "Please set your API key using ONE of these methods:\n\n"
            "METHOD 1 - Environment Variable (Recommended):\n"
            "  Windows: set GEMINI_API_KEY=your_api_key_here\n"
            "  Linux/Mac: export GEMINI_API_KEY=your_api_key_here\n\n"
            "METHOD 2 - Direct in Script:\n"
            "  Edit line ~23 in the script:\n"
            "  GEMINI_API_KEY = \"your_api_key_here\"\n\n"
            "METHOD 3 - Create .env file:\n"
            "  Create a file named '.env' with:\n"
            "  GEMINI_API_KEY=your_api_key_here\n\n"
            "Get your API key from: https://aistudio.google.com/apikey\n"
            "="*70
        )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        logging.info(f"✓ Gemini API key configured successfully")
        return model
    except Exception as e:
        raise ValueError(f"Failed to initialize Gemini with provided API key: {e}")

# =============== IMAGE PROCESSING ===============
def is_valid_image(image_data, min_kb=10, min_dim=100):
    """Check if image meets size and dimension requirements"""
    try:
        size_kb = len(image_data) / 1024
        if size_kb < min_kb:
            return False, f"Too small ({size_kb:.1f}KB)"
        
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size
        
        if width < min_dim or height < min_dim:
            return False, f"Dimensions too small ({width}x{height})"
        
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 20:
            return False, f"Extreme aspect ratio"
        
        return True, f"Valid ({width}x{height}, {size_kb:.1f}KB)"
    
    except Exception as e:
        return False, f"Invalid: {str(e)}"

def optimize_image(image_data, ext):
    """Optimize image by resizing and compressing"""
    try:
        img = Image.open(io.BytesIO(image_data))
        width, height = img.size
        
        if Config.RESIZE_LARGE_IMAGES and max(width, height) > Config.MAX_IMAGE_DIMENSION:
            ratio = Config.MAX_IMAGE_DIMENSION / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        output = io.BytesIO()
        
        if ext.lower() in ['jpg', 'jpeg']:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(output, format='JPEG', quality=Config.COMPRESSION_QUALITY, optimize=True)
        elif ext.lower() == 'png':
            img.save(output, format='PNG', optimize=True)
        else:
            img.save(output, format=img.format or 'PNG')
        
        return output.getvalue()
    
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        return image_data

# =============== AI IDENTIFICATION ===============
def identify_image_with_gemini(model, image_data: bytes, page_num: int, img_index: int) -> Dict:
    """Use Gemini Vision to identify what the image contains"""
    
    prompt = """Analyze this technical image from a tractor manual and provide:

1. PRIMARY CATEGORY (choose ONE most appropriate):
   - battery, engine, fuel_tank, transmission, hydraulics, electrical, cooling_system,
   - exhaust, filters, wheels_tires, steering, brakes, lights, controls, seats, cabin,
   - pto, hitch, frame_chassis, diagram, schematic, unknown

2. COMPONENT NAME: Specific part name (e.g., "12V Battery", "Fuel Filter", "Hydraulic Pump")

3. DESCRIPTION: Brief description (1-2 sentences)

4. TAGS: Additional relevant keywords (comma-separated)

Respond ONLY in this JSON format:
{
  "category": "category_name",
  "component_name": "Specific Part Name",
  "description": "Brief description",
  "tags": ["tag1", "tag2", "tag3"],
  "confidence": "high/medium/low"
}"""
    
    for attempt in range(Config.AI_RETRY_ATTEMPTS):
        try:
            # Convert image to PIL for Gemini
            img = Image.open(io.BytesIO(image_data))
            
            # Generate response
            response = model.generate_content([prompt, img])
            
            # Parse JSON from response
            response_text = response.text.strip()
            
            # Clean up response (remove markdown code blocks if present)
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            result = json.loads(response_text)
            
            # Validate category
            category = result.get('category', 'unknown').lower().replace(' ', '_')
            if category not in Config.EXPECTED_CATEGORIES:
                category = 'unknown'
            
            result['category'] = category
            result['ai_model'] = Config.GEMINI_MODEL
            result['processing_attempt'] = attempt + 1
            
            logging.info(f"✓ AI identified: {result['component_name']} ({category})")
            return result
        
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error (429)
            if '429' in error_msg or 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower():
                # Extract retry delay from error message if available
                retry_delay = Config.AI_RETRY_DELAY
                if 'retry_delay' in error_msg:
                    try:
                        import re
                        delay_match = re.search(r'seconds: (\d+)', error_msg)
                        if delay_match:
                            retry_delay = int(delay_match.group(1)) + 2  # Add 2 seconds buffer
                    except:
                        pass
                
                logging.warning(f"⚠️ Rate limit hit (attempt {attempt + 1}/{Config.AI_RETRY_ATTEMPTS}). Waiting {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            
            # For JSON parse errors
            if isinstance(e, json.JSONDecodeError):
                logging.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
                if attempt < Config.AI_RETRY_ATTEMPTS - 1:
                    time.sleep(Config.AI_RETRY_DELAY)
                    continue
            
            # For other errors
            logging.error(f"AI identification error (attempt {attempt + 1}): {e}")
            if attempt < Config.AI_RETRY_ATTEMPTS - 1:
                time.sleep(Config.AI_RETRY_DELAY)
            else:
                # Fallback response after all retries
                return {
                    'category': 'unknown',
                    'component_name': f'Page{page_num}_Image{img_index}',
                    'description': 'AI identification failed after retries',
                    'tags': [],
                    'confidence': 'low',
                    'error': error_msg[:200]  # Truncate long error messages
                }
    
    return {
        'category': 'unknown',
        'component_name': f'Page{page_num}_Image{img_index}',
        'description': 'Max retries reached',
        'tags': [],
        'confidence': 'low'
    }

def generate_smart_filename(page_num: int, img_index: int, ai_result: Dict, ext: str, image_data: bytes) -> str:
    """Generate descriptive filename based on AI identification"""
    category = ai_result.get('category', 'unknown')
    component = ai_result.get('component_name', 'component')
    
    # Clean component name for filename
    component_clean = component.replace(' ', '_').replace('/', '_').replace('\\', '_')
    component_clean = ''.join(c for c in component_clean if c.isalnum() or c == '_')[:50]
    
    # Add hash for uniqueness
    content_hash = hashlib.md5(image_data).hexdigest()[:6]
    
    ext = ext.lower().replace('jpeg', 'jpg')
    
    # Format: battery_12v_battery_p001_i01_abc123.jpg
    return f"{category}_{component_clean}_p{page_num:03d}_i{img_index:02d}_{content_hash}.{ext}"

# =============== MAIN EXTRACTION ===============
def extract_and_categorize_images():
    """Main function with AI-powered categorization"""
    logger = setup_logging()
    
    # Validate PDF
    if not os.path.exists(Config.PDF_PATH):
        logger.error(f"PDF not found: {Config.PDF_PATH}")
        return False
    
    # Setup Gemini
    try:
        model = setup_gemini()
        logger.info(f"✓ Gemini AI initialized: {Config.GEMINI_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        return False
    
    # Create organized output structure
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = Path(Config.OUTPUT_ROOT) / Path(Config.PDF_PATH).stem / today
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create category folders
    for category in Config.EXPECTED_CATEGORIES:
        (output_folder / category).mkdir(exist_ok=True)
    
    # Metadata tracking
    extraction_metadata = {
        'pdf_path': Config.PDF_PATH,
        'extraction_date': datetime.now().isoformat(),
        'ai_model': Config.GEMINI_MODEL,
        'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('_') and k != 'GEMINI_API_KEY'},
        'images': [],
        'categories_summary': {}
    }
    
    try:
        doc = fitz.open(Config.PDF_PATH)
        total_pages = len(doc)
        logger.info(f"📄 PDF: {Config.PDF_PATH} ({total_pages} pages)")
        logger.info(f"📁 Output: {output_folder}")
        
        images_extracted = 0
        images_skipped = 0
        category_counts = {cat: 0 for cat in Config.EXPECTED_CATEGORIES}
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Page {page_num + 1}/{total_pages}: {len(image_list)} images found")
            logger.info(f"{'='*60}")
            
            if len(image_list) > Config.MAX_IMAGES_PER_PAGE:
                image_list = image_list[:Config.MAX_IMAGES_PER_PAGE]
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    ext = base_image["ext"].lower()
                    
                    if ext not in Config.SUPPORTED_FORMATS:
                        images_skipped += 1
                        continue
                    
                    # Validate image
                    is_valid, message = is_valid_image(image_bytes, Config.MIN_IMAGE_SIZE_KB, Config.MIN_DIMENSION)
                    
                    if not is_valid:
                        logger.debug(f"⊗ Skipped: {message}")
                        images_skipped += 1
                        continue
                    
                    # Optimize
                    optimized_bytes = optimize_image(image_bytes, ext)
                    
                    # AI IDENTIFICATION
                    logger.info(f"\n🤖 Analyzing image {img_index + 1} with Gemini AI...")
                    ai_result = identify_image_with_gemini(model, optimized_bytes, page_num + 1, img_index + 1)
                    
                    # Rate limiting
                    time.sleep(Config.RATE_LIMIT_DELAY)
                    
                    # Generate smart filename
                    filename = generate_smart_filename(page_num + 1, img_index + 1, ai_result, ext, optimized_bytes)
                    
                    # Save to category folder
                    category = ai_result['category']
                    category_folder = output_folder / category
                    filepath = category_folder / filename
                    
                    with open(filepath, "wb") as f:
                        f.write(optimized_bytes)
                    
                    # Update metadata
                    img_metadata = {
                        'filename': filename,
                        'category': category,
                        'page': page_num + 1,
                        'index': img_index + 1,
                        'format': ext,
                        'size_kb': len(optimized_bytes) / 1024,
                        'ai_identification': ai_result,
                        'filepath': str(filepath.relative_to(Config.OUTPUT_ROOT))
                    }
                    extraction_metadata['images'].append(img_metadata)
                    
                    category_counts[category] += 1
                    images_extracted += 1
                    
                    logger.info(f"✓ SAVED: {filename}")
                    logger.info(f"  └─ Category: {category}")
                    logger.info(f"  └─ Component: {ai_result['component_name']}")
                    logger.info(f"  └─ Confidence: {ai_result.get('confidence', 'unknown')}")
                
                except Exception as e:
                    logger.error(f"✗ Error processing image {img_index + 1}: {e}")
                    images_skipped += 1
        
        # Save metadata
        extraction_metadata['categories_summary'] = category_counts
        metadata_path = output_folder / Config.METADATA_FILE
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_metadata, f, indent=2, default=str)
        
        # Generate categorized index
        generate_categorized_index(output_folder, category_counts)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("EXTRACTION COMPLETE")
        logger.info("="*70)
        logger.info(f"✓ PDF: {Path(Config.PDF_PATH).name}")
        logger.info(f"✓ Pages: {total_pages}")
        logger.info(f"✓ Images extracted: {images_extracted}")
        logger.info(f"✓ Images skipped: {images_skipped}")
        logger.info(f"✓ Output: {output_folder}")
        logger.info("\n📊 CATEGORY BREAKDOWN:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                logger.info(f"   • {category:20s}: {count:3d} images")
        logger.info("="*70)
        
        doc.close()
        return True
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return False

# =============== HTML INDEX GENERATION ===============
def generate_categorized_index(output_folder: Path, category_counts: Dict):
    """Generate HTML index organized by categories"""
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>AI-Categorized Images - {output_folder.parent.name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                  gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center; 
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 32px; font-weight: bold; color: #667eea; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .category-section {{ background: white; margin: 20px 0; padding: 20px; 
                             border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .category-header {{ font-size: 24px; font-weight: bold; color: #333; 
                            margin-bottom: 20px; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); 
                       gap: 20px; }}
        .image-card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; 
                       background: #fafafa; transition: transform 0.2s, box-shadow 0.2s; }}
        .image-card:hover {{ transform: translateY(-5px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
        .image-card img {{ width: 100%; height: 200px; object-fit: contain; 
                           border-radius: 5px; background: white; }}
        .component-name {{ font-weight: bold; color: #333; margin: 10px 0 5px 0; 
                           font-size: 14px; }}
        .description {{ font-size: 12px; color: #666; margin: 5px 0; }}
        .tags {{ margin-top: 10px; }}
        .tag {{ display: inline-block; background: #e8f4fd; color: #0066cc; 
                padding: 3px 8px; border-radius: 3px; font-size: 11px; margin: 2px; }}
        .confidence {{ float: right; padding: 3px 8px; border-radius: 3px; font-size: 11px; }}
        .confidence-high {{ background: #d4edda; color: #155724; }}
        .confidence-medium {{ background: #fff3cd; color: #856404; }}
        .confidence-low {{ background: #f8d7da; color: #721c24; }}
        .metadata {{ font-size: 11px; color: #999; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 AI-Categorized Tractor Parts</h1>
        <p>Powered by {Config.GEMINI_MODEL}</p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">{sum(category_counts.values())}</div>
            <div class="stat-label">Total Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{sum(1 for c in category_counts.values() if c > 0)}</div>
            <div class="stat-label">Categories Found</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{Path(Config.PDF_PATH).name}</div>
            <div class="stat-label">Source PDF</div>
        </div>
    </div>
"""
    
    # Load metadata for detailed information
    metadata_path = output_folder / Config.METADATA_FILE
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        images_by_category = {}
        for img in metadata['images']:
            cat = img['category']
            if cat not in images_by_category:
                images_by_category[cat] = []
            images_by_category[cat].append(img)
    else:
        images_by_category = {}
    
    # Generate sections for each category
    for category in sorted(category_counts.keys(), key=lambda x: category_counts[x], reverse=True):
        if category_counts[category] == 0:
            continue
        
        html_content += f"""
    <div class="category-section">
        <div class="category-header">
            {category.replace('_', ' ').title()} ({category_counts[category]} images)
        </div>
        <div class="image-grid">
"""
        
        if category in images_by_category:
            for img in images_by_category[category]:
                ai_info = img.get('ai_identification', {})
                confidence = ai_info.get('confidence', 'unknown')
                
                html_content += f"""
            <div class="image-card">
                <img src="{img['category']}/{img['filename']}" alt="{img['filename']}">
                <div class="component-name">
                    {ai_info.get('component_name', 'Unknown Component')}
                    <span class="confidence confidence-{confidence}">{confidence}</span>
                </div>
                <div class="description">{ai_info.get('description', 'No description')}</div>
                <div class="tags">
"""
                
                for tag in ai_info.get('tags', []):
                    html_content += f'<span class="tag">{tag}</span>'
                
                html_content += f"""
                </div>
                <div class="metadata">
                    Page {img['page']} • {img['size_kb']:.1f} KB • {img['format'].upper()}
                </div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    index_path = output_folder / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"✓ HTML index generated: {index_path}")
    return index_path

# =============== MAIN EXECUTION ===============
if __name__ == "__main__":
    print("="*70)
    print("🤖 AI-Powered PDF Image Extractor & Categorizer")
    print("="*70)
    print(f"PDF: {Config.PDF_PATH}")
    print(f"AI Model: {Config.GEMINI_MODEL}")
    print(f"Output: {Config.OUTPUT_ROOT}")
    print("="*70)
    
    success = extract_and_categorize_images()
    
    if success:
        print("\n✅ SUCCESS! Check the output folder for categorized images.")
    else:
        print("\n❌ FAILED! Check the log file for details.")
    
    print("\nPress Enter to exit...")
    input()