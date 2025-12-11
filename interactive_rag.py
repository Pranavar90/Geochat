import sys
from rag_engine_faiss import RAGEngine
import time

def main():
    print("=" * 60)
    print("          🤖 GeoChat Interactive Terminal 🤖")
    print("WARNING: Ensure 'ollama serve' is running in another terminal.")
    print("=" * 60)
    
    # Initialize Engine
    print("\nInitializing RAG Engine... (Loading FAISS & Models)")
    try:
        start_time = time.time()
        engine = RAGEngine()
        elapsed = time.time() - start_time
        print(f"✅ Engine loaded in {elapsed:.2f}s")
        print(f"📚 Knowledge Base: {engine.index.ntotal} chunks available")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Could not load RAGEngine.\n{e}")
        return

    print("\n" + "-" * 60)
    print("Instructions:")
    print(" - Type your question and press ENTER.")
    print(" - Type 'exit', 'quit', or 'q' to close the program.")
    print("-" * 60 + "\n")

    # Chat Loop
    while True:
        try:
            user_input = input("USER >> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nExiting GeoChat. Goodbye! 👋")
                break
            
            print("\nGeoChat is thinking...", end="\r")
            response = engine.answer(user_input)
            
            # Clear "thinking" line and print response
            print(" " * 30, end="\r") 
            print("GEOCHAT >>")
            print(response)
            print("\n" + "-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user. Exiting...")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ Error during processing: {e}\n")

if __name__ == "__main__":
    main()
