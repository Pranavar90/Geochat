import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import pickle

# Configuration
CHUNKS_DIR = "geochat_data/chunks"
DB_DIR = "geochat_data/vector_db"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def embed_and_store():
    # 1. Initialize Model
    print(f"Loading Model: {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    embedding_dim = model.get_sentence_embedding_dimension()
    
    # 2. Load Chunks
    print("Loading chunks from disk...")
    chunk_files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith('.json')]
    
    if not chunk_files:
        print("No chunks found! Run process_pdfs.py first.")
        return

    print(f"Found {len(chunk_files)} chunks. Processing...")
    
    # Lists to store everything
    all_embeddings = []
    all_metadata = []
    all_ids = []
    all_texts = []
    
    # Process all chunks
    BATCH_SIZE = 128
    current_batch_texts = []
    current_batch_metas = []
    current_batch_ids = []
    current_batch_orig_texts = []
    
    for i, filename in enumerate(tqdm(chunk_files, desc="Embedding", unit="chunk")):
        filepath = os.path.join(CHUNKS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            chunk = json.load(f)
            
        text_content = chunk['text']
        
        current_batch_texts.append(text_content)
        current_batch_ids.append(chunk['chunk_id'])
        current_batch_orig_texts.append(text_content)
        current_batch_metas.append({
            "book": chunk['book'],
            "chapter": chunk['chapter'],
            "section": chunk['section'],
            "source": filename
        })
        
        # Process batch
        if len(current_batch_texts) >= BATCH_SIZE or i == len(chunk_files) - 1:
            # Embed the batch
            embeddings = model.encode(current_batch_texts, convert_to_tensor=False)
            
            # Store
            all_embeddings.extend(embeddings)
            all_metadata.extend(current_batch_metas)
            all_ids.extend(current_batch_ids)
            all_texts.extend(current_batch_orig_texts)
            
            # Reset batch
            current_batch_texts = []
            current_batch_metas = []
            current_batch_ids = []
            current_batch_orig_texts = []
    
    # 3. Create FAISS Index
    print("Creating FAISS index...")
    embeddings_np = np.array(all_embeddings).astype('float32')
    
    # Use IndexFlatL2 for exact search (or IndexIVFFlat for faster approximate search)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    
    # 4. Save everything
    os.makedirs(DB_DIR, exist_ok=True)
    
    print("Saving index and metadata...")
    faiss.write_index(index, os.path.join(DB_DIR, "faiss.index"))
    
    # Save metadata separately
    with open(os.path.join(DB_DIR, "metadata.pkl"), 'wb') as f:
        pickle.dump({
            'ids': all_ids,
            'metadata': all_metadata,
            'texts': all_texts
        }, f)
    
    print("-" * 30)
    print("SUCCESS")
    print(f"Processed: {len(chunk_files)} chunks")
    print(f"Vector Dimension: {embedding_dim}")
    print(f"Total Documents in DB: {index.ntotal}")
    print(f"DB Path: {os.path.abspath(DB_DIR)}")
    print("-" * 30)

if __name__ == "__main__":
    embed_and_store()
