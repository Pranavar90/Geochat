import pypdf
import os

pdf_path = "geochat_data/raw_pdfs/Geology_of_india_by_dn_wadia.pdf"

try:
    reader = pypdf.PdfReader(pdf_path)
    page = reader.pages[10] # Try a page that likely has text (Preface was page viii, so ~10)
    text = page.extract_text()
    print("--- RAW TEXT START ---")
    print(text[:1000])
    print("--- RAW TEXT END ---")
except Exception as e:
    print(f"Error: {e}")
