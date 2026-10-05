import os
import asyncio
import io
from pathlib import Path
from pdf2image import convert_from_path

# Import your rotating vision client
from utils.vision_client import vision_call

OUTPUT_DIR = Path("txt_knowledge")
OUTPUT_FILE = "rotavator_manual_extracted.txt"
MACHINE_TYPE = "rotavator"

SMART_OCR_PROMPT = f"""
You are an expert technical writer digitizing an old, scanned agricultural repair manual for a {MACHINE_TYPE}.
Look at this page image and extract all information into structured text.

RULES:
1. Extract all text exactly as written. Do not summarize or paraphrase.
2. Figure out the main mechanical task on this page and write it in the PROBLEM field.
3. Group instructions logically under STEPS:.
4. If there is a diagram or illustration, create a section called DIAGRAM PARTS: and list every labeled part and its reference number.

Format your response STRICTLY like this, with no markdown code blocks around it:

PROBLEM: [Identify the main task, e.g., Dismantling the Clutch Unit]
MACHINE: {MACHINE_TYPE}
SYMPTOM: [What symptom would require this task? Guess briefly based on the parts]
STEPS:
[Extract the steps here verbatim]

DIAGRAM PARTS:
[List the parts and numbers here, e.g., clutch plate fixed (77)]
"""

async def process_page(image_bytes: bytes, page_num: int, max_retries: int = 3) -> str:
    print(f"   🤖 Sending page {page_num} to Gemini Vision...")
    for attempt in range(max_retries):
        try:
            text = await vision_call(
                prompt=SMART_OCR_PROMPT,
                image_bytes=image_bytes,
                max_tokens=1500,
                temperature=0.1,
            )
            # Clean up markdown
            text = text.replace("```text", "").replace("```", "").replace("```json", "").strip()
            return f"🔹 Chunk {page_num}\n{text}\n\n"
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 5
                print(f"   ⚠️ Page {page_num} failed (attempt {attempt+1}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"   ❌ Page {page_num} failed after {max_retries} attempts: {e}")
                return f"🔹 Chunk {page_num}\nPROBLEM: [OCR FAILED]\nMACHINE: {MACHINE_TYPE}\n\n"

async def main():
    pdf_path = input("Enter path to scanned PDF: ").strip()
    # Strip quotes if dragged-and-dropped into terminal
    pdf_path = pdf_path.strip('"').strip("'") 
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"\n📄 Slicing PDF into images: {pdf_path}")
    poppler_dir = r"D:\poppler-26.02.0\Library\bin" 
    images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_dir)
    print(f"   {len(images)} pages extracted.\n")
    
    output_path = OUTPUT_DIR / OUTPUT_FILE
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# AgriFix Knowledge Base — {MACHINE_TYPE.upper()} (Gemini Vision OCR)\n\n")
        
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()
            
            chunk_text = await process_page(img_bytes, i + 1)
            
            if chunk_text:
                f.write(chunk_text)
                print(f"   ✅ Page {i + 1} processed and saved.")
            
            # Rate limit buffer
            await asyncio.sleep(2)
            
    print(f"\n🎉 SUCCESS! Saved {len(images)} pages to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())