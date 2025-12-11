import torch
import time
from sentence_transformers import SentenceTransformer

def check_gpu():
    print("-" * 30)
    print("      GPU DIAGNOSTIC TEST      ")
    print("-" * 30)
    
    # 1. Check Torch CUDA availability
    is_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")
    
    if is_available:
        print(f"Device Name:    {torch.cuda.get_device_name(0)}")
        print(f"Device Count:   {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("❌ WARNING: PyTorch cannot see your GPU. It will use CPU (Slow).")

    # 2. Test Embedding Speed
    print("\nLoading Model to test speed...")
    start_load = time.time()
    device = "cuda" if is_available else "cpu"
    try:
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Model loaded in {time.time() - start_load:.2f}s on {device.upper()}")
    
    print("\nRunning dummy embedding (100 sentences)...")
    sentences = ["This is a test sentence for speed check."] * 100
    
    start_embed = time.time()
    model.encode(sentences)
    total_time = time.time() - start_embed
    
    print(f"Time taken: {total_time:.4f}s")
    print(f"Speed:      {100/total_time:.2f} sentences/second")
    
    if is_available and total_time < 2.0:
        print("\n✅ PASSED: GPU is working correctly.")
        print("You can run 'embed_chunks.py' safely. It should take < 10 mins.")
    elif not is_available:
        print("\n❌ FAILED: Using CPU.")
    else:
        print("\n⚠️ WARNING: GPU detected but seems slow.")

if __name__ == "__main__":
    check_gpu()
