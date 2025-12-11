import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import time

# Configuration
CHUNKS_DIR = "geochat_data/chunks"
DB_DIR = "geochat_data/vector_db"
COLLECTION_NAME = "geochat_docs"
# Switched from Nomic to MiniLM due to GPU stability issues (4GB VRAM limitation)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def embed_and_store():
    # 1. Initialize Model
    print(f"Loading Model: {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    
    # 2. Initialize ChromaDB
    print(f"Initializing ChromaDB at {DB_DIR}...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # 3. Load Chunks
    print("Loading chunks from disk...")
    chunk_files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith('.json')]
    
    if not chunk_files:
        print("No chunks found! Run process_pdfs.py first.")
        return

    documents = []
    metadatas = []
    ids = []
    
    # Batch processing to manage memory
    # Reduced to 32 to prevent ChromaDB compaction errors on Windows
    BATCH_SIZE = 32 
    
    print(f"Found {len(chunk_files)} chunks. Processing...")
    
    current_batch_files = []
    
    # Using tqdm again for specific progress
    for i, filename in enumerate(tqdm(chunk_files, desc="Processing", unit="chunk")):
        filepath = os.path.join(CHUNKS_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            chunk = json.load(f)
            
        # MiniLM doesn't need prefixes
        text_content = chunk['text']
        
        current_batch_files.append((
            chunk['chunk_id'],
            text_content,
            {
                "book": chunk['book'],
                "chapter": chunk['chapter'],
                "section": chunk['section'],
                "source": filename
            }
        ))
        
        # Process batch
        if len(current_batch_files) >= BATCH_SIZE or i == len(chunk_files) - 1:
            batch_ids = [item[0] for item in current_batch_files]
            batch_texts = [item[1] for item in current_batch_files]
            batch_metas = [item[2] for item in current_batch_files]
            
            # Embed
            embeddings = model.encode(batch_texts, convert_to_tensor=False)
            
            # Add to Chroma
            collection.upsert(
                ids=batch_ids,
                documents=batch_texts, 
                metadatas=batch_metas,
                embeddings=embeddings.tolist()
            )
            
            # Small delay to prevent ChromaDB compaction errors
            time.sleep(0.05)
            
            current_batch_files = []

    count = collection.count()
    print("-" * 30)
    print("SUCCESS")
    print(f"Processed: {len(chunk_files)} chunks")
    print(f"Vector Dimension: {model.get_sentence_embedding_dimension()}")
    print(f"Total Documents in DB: {count}")
    print(f"DB Path: {os.path.abspath(DB_DIR)}")
    print("-" * 30)

if __name__ == "__main__":
    embed_and_store()
