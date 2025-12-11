import os
import re
import json
import glob
import random
from typing import List, Dict, Any

try:
    import pdfplumber
except ImportError:
    print("Error: pdfplumber is not installed. Please run: pip install pdfplumber")
    exit(1)

try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("Warning: OCR dependencies missing. OCR fallback will not work.")
    print("Run: pip install pytesseract pdf2image Pillow")

# Configuration
INPUT_DIR = "geochat_data/raw_pdfs"
CLEAN_TEXT_DIR = "geochat_data/clean_text"
CHUNKS_DIR = "geochat_data/chunks"
REPORT_FILE = "geochat_data/verification_report.md"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 1400

# OCR Configuration (Windows paths - Adjust if needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Regex Patterns
HEADER_FOOTER_PATTERN = re.compile(r'^\d+\s*$', re.MULTILINE) 
HYPHEN_FIX_PATTERN = re.compile(r'(\w+)-\n(\w+)')
MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')

def clean_text(text: str) -> str:
    if not text: return ""
    text = HEADER_FOOTER_PATTERN.sub('', text)
    text = HYPHEN_FIX_PATTERN.sub(r'\1\2', text)
    text = MULTIPLE_NEWLINES.sub('\n\n', text)
    return text.strip()

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def split_into_sentences(text: str) -> List[str]:
    return SENTENCE_ENDINGS.split(text)

def chunk_text(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk_tokens = 0
    current_chunk_text = [] # list of strings
    chunk_counter = 0

    def finalize_chunk(chunk_text_list, next_overlap_text=[]):
        nonlocal chunk_counter
        chunk_str = "\n\n".join(chunk_text_list)
        chunk_id = f"{metadata['book_id']}_chunk_{chunk_counter:04d}"
        chunks.append({
            "chunk_id": chunk_id,
            "book": metadata['book'],
            "chapter": "Unknown",
            "section": "Unknown",
            "text": chunk_str
        })
        chunk_counter += 1
        return next_overlap_text, estimate_tokens("\n\n".join(next_overlap_text)) if next_overlap_text else 0

    for para in paragraphs:
        para = para.strip()
        if not para: continue
        
        para_tokens = estimate_tokens(para)
        
        if para_tokens > MAX_CHUNK_SIZE:
             sentences = split_into_sentences(para)
             for sent in sentences:
                 sent_tokens = estimate_tokens(sent)
                 if current_chunk_tokens + sent_tokens > CHUNK_SIZE:
                     if current_chunk_tokens >= MIN_CHUNK_SIZE:
                         current_chunk_text, current_chunk_tokens = finalize_chunk(current_chunk_text)
                 current_chunk_text.append(sent)
                 current_chunk_tokens += sent_tokens
             continue

        if current_chunk_tokens + para_tokens > CHUNK_SIZE:
             if current_chunk_tokens >= MIN_CHUNK_SIZE:
                 overlap_text = []
                 if current_chunk_text:
                     last_item = current_chunk_text[-1]
                     if estimate_tokens(last_item) < CHUNK_OVERLAP:
                         overlap_text = [last_item]
                 current_chunk_text, current_chunk_tokens = finalize_chunk(current_chunk_text, overlap_text)
        
        current_chunk_text.append(para)
        current_chunk_tokens += para_tokens
        
    if current_chunk_text and current_chunk_tokens > 0:
        finalize_chunk(current_chunk_text)
    return chunks

def extract_text_with_fallback(filepath: str) -> Dict[str, Any]:
    """Extracts text, falling back to OCR if a page is empty."""
    full_text = ""
    pages_ocr_count = 0
    
    try:
        # 1. Try Standard Extraction
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text(x_tolerance=1)
                
                # Check for empty page -> Trigger OCR
                if not text or len(text.strip()) < 50:
                    try:
                        # Convert specific page to image
                        images = convert_from_path(filepath, first_page=i+1, last_page=i+1, poppler_path=r'C:\Program Files\poppler-24.02.0\Library\bin') # Attempt default path
                        if images:
                            ocr_text = pytesseract.image_to_string(images[0])
                            full_text += ocr_text + "\n"
                            pages_ocr_count += 1
                    except Exception as e:
                         # Fallback failed (missing poppler/tesseract?)
                         pass # Keep going
                else:
                    full_text += text + "\n"
                    
    except Exception as e:
        return {"error": str(e), "text": ""}
        
    return {
        "text": full_text, 
        "ocr_pages": pages_ocr_count, 
        "total_pages": total_pages if 'total_pages' in locals() else 0
    }

def process_pdfs():
    if not os.path.exists(CLEAN_TEXT_DIR): os.makedirs(CLEAN_TEXT_DIR)
    if not os.path.exists(CHUNKS_DIR): os.makedirs(CHUNKS_DIR)
    
    pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    report_lines = ["# PDF Verification Report\n"]
    
    print(f"Found {len(pdf_files)} PDFs. Processing...")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}")
        
        # ID Generation
        safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', filename.split('.')[0])
        book_id = safe_id[:15]
        
        # Extraction
        result = extract_text_with_fallback(pdf_path)
        
        if "error" in result:
             print(f"  - FAILED: {result['error']}")
             report_lines.append(f"## {filename}\n**Status**: FAILED ❌\n**Error**: {result['error']}\n")
             continue
             
        text = result["text"]
        
        # Validation
        if len(text.strip()) < 100:
            print(f"  - EMPTY: Less than 100 chars extracted.")
            report_lines.append(f"## {filename}\n**Status**: EMPTY/FAILED ❌\n**Extracted**: {len(text)} chars\n**Note**: OCR might be missing or failed.\n")
            continue
            
        print(f"  - Extracted: {len(text)} chars (OCR used on {result['ocr_pages']} pages)")
        
        # Save Clean Text
        clean_path = os.path.join(CLEAN_TEXT_DIR, f"{filename}_clean.txt")
        cleaned_text = clean_text(text)
        with open(clean_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
            
        # Chunking
        metadata = {"book": filename.replace('.pdf', ''), "book_id": book_id}
        chunks = chunk_text(cleaned_text, metadata)
        
        for chunk in chunks:
            with open(os.path.join(CHUNKS_DIR, f"{chunk['chunk_id']}.json"), 'w', encoding='utf-8') as f:
                json.dump(chunk, f, indent=2)
                
        # Update Report
        status_icon = "⚠️" if result['ocr_pages'] > 0 else "✅"
        sample_chunk = random.choice(chunks) if chunks else {"text": "No chunks generated"}
        
        report_lines.append(f"## {filename}\n")
        report_lines.append(f"**Status**: SUCCESS {status_icon}\n")
        report_lines.append(f"- **Total Pages**: {result['total_pages']}\n")
        report_lines.append(f"- **OCR Used**: {result['ocr_pages']} pages\n")
        report_lines.append(f"- **Total Chunks**: {len(chunks)}\n")
        report_lines.append(f"### Text Sample (First 500 chars)\n")
        report_lines.append(f"```text\n{cleaned_text[:500]}\n```\n")
        report_lines.append(f"### Chunk Sample (ID: {sample_chunk.get('chunk_id')})\n")
        report_lines.append(f"```json\n{json.dumps(sample_chunk, indent=2)}\n```\n")
        report_lines.append(f"---\n")

    # Save verification report
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    print(f"Report verified: {REPORT_FILE}")

if __name__ == "__main__":
    process_pdfs()
