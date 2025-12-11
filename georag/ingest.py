import os
import fitz  # PyMuPDF
import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from . import config, utils

logger = utils.setup_logger("Ingest")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_data = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        clean_content = utils.clean_text(text)
        if clean_content:
            text_data.append({
                "text": clean_content,
                "page": page_num + 1,
                "source": os.path.basename(pdf_path)
            })
    return text_data

def chunk_text(text_data, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Chunks text data into smaller pieces."""
    chunks = []
    for item in text_data:
        text = item['text']
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": item['source'],
                    "page": item['page']
                }
            })
            if i + chunk_size >= len(words):
                break
    return chunks

def ingest_data():
    """Main ingestion function."""
    # 1. Load Data
    all_chunks = []
    pdf_files = [f for f in os.listdir(config.DATA_DIR) if f.endswith('.pdf')]
    
    if not pdf_files:
        logger.error("No PDF files found in data directory!")
        return

    logger.info(f"Found {len(pdf_files)} PDFs: {pdf_files}")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(config.DATA_DIR, pdf_file)
        logger.info(f"Processing {pdf_file}...")
        raw_text = extract_text_from_pdf(pdf_path)
        file_chunks = chunk_text(raw_text)
        all_chunks.extend(file_chunks)
        
    logger.info(f"Total chunks created: {len(all_chunks)}")
    
    # 2. Embed
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE)
    
    texts = [c['text'] for c in all_chunks]
    logger.info("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # 3. Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 4. Save
    logger.info(f"Saving index to {config.FAISS_INDEX_PATH}")
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    
    with open(config.METADATA_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)
        
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
