import pdfplumber
import os

pdf_path = "geochat_data/raw_pdfs/Geology_of_india_by_dn_wadia.pdf"

try:
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        print(f"Total Pages: {num_pages}")
        
        text_pages = 0
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                text_pages += 1
            if i % 50 == 0: print(f"Checked {i} pages...")
            
        print(f"Pages with text: {text_pages}")
        if text_pages < num_pages * 0.5:
            print("CONCLUSION: MOSTLY SCANNED IMAGES")
        else:
            print("CONCLUSION: MOSTLY TEXT")

except Exception as e:
    print(f"Error: {e}")
