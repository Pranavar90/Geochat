import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import torch

# Configuration
DB_DIR = "geochat_data/vector_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def smoke_test():
    print(f"Loading Model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    
    print(f"Loading FAISS index from {DB_DIR}...")
    index = faiss.read_index(os.path.join(DB_DIR, "faiss.index"))
    
    with open(os.path.join(DB_DIR, "metadata.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    print(f"Index loaded: {index.ntotal} vectors")
    
    while True:
        query = input("\nEnter query (or 'q' to quit): ").strip()
        if query.lower() in ('q', 'quit', 'exit'):
            break
            
        # Embed query
        query_vec = model.encode([query]).astype('float32')
        
        # Search
        distances, indices = index.search(query_vec, k=3)
        
        print(f"\nTop 3 Results for '{query}':")
        for i, idx in enumerate(indices[0]):
            chunk_id = data['ids'][idx]
            dist = distances[0][i]
            meta = data['metadata'][idx]
            doc = data['texts'][idx]
            
            display_text = doc[:300] + "..."
            
            print(f"[{i+1}] ID: {chunk_id} | Dist: {dist:.4f}")
            print(f"    Source: {meta.get('book')} | Chapter: {meta.get('chapter')}")
            print(f"    Preview: {display_text}")
            print("-" * 40)

if __name__ == "__main__":
    smoke_test()
