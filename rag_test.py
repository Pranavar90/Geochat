import chromadb
from sentence_transformers import SentenceTransformer
import torch

# Configuration
DB_DIR = "geochat_data/vector_db"
COLLECTION_NAME = "geochat_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def smoke_test():
    print(f"Loading Model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=DEVICE)
    
    print(f"Connecting to DB at {DB_DIR}...")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("Error: Collection not found! Run embed_chunks.py first.")
        return

    print(f"Collection count: {collection.count()}")
    
    while True:
        query = input("\nEnter query (or 'q' to quit): ").strip()
        if query.lower() in ('q', 'quit', 'exit'):
            break
            
        # MiniLM doesn't need prefixes
        query_vec = model.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=3
        )
        
        print(f"\nTop 3 Results for '{query}':")
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            dist = results['distances'][0][i]
            meta = results['metadatas'][0][i]
            doc = results['documents'][0][i]
            
            # Display text (no prefixes to remove)
            display_text = doc[:300] + "..."
            
            print(f"[{i+1}] ID: {chunk_id} | Dist: {dist:.4f}")
            print(f"    Source: {meta.get('book')} | Chapter: {meta.get('chapter')}")
            print(f"    Preview: {display_text}")
            print("-" * 40)

if __name__ == "__main__":
    smoke_test()
