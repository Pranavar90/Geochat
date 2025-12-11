import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_NAME = "phi3:mini"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Chunking
CHUNK_SIZE = 500  # tokens approx
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 5
