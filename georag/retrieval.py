import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from . import config, utils

logger = utils.setup_logger("Retrieval")

class Retriever:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.model = None
        self.load_index()

    def load_index(self):
        """Loads FAISS index and metadata."""
        try:
            logger.info(f"Loading FAISS index from {config.FAISS_INDEX_PATH}")
            self.index = faiss.read_index(config.FAISS_INDEX_PATH)
            
            with open(config.METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
                
            logger.info("Loading embedding model...")
            self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE)
            logger.info("Retriever ready.")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def retrieve(self, query, k=config.TOP_K):
        """Retrieves top-k relevant chunks."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                item = self.metadata[idx]
                results.append({
                    "text": item['text'],
                    "metadata": item['metadata'],
                    "distance": float(dist)
                })
        return results
