from rag_engine import RAGEngine
import sys

def main():
    print("------------------------------------------------")
    print("          🌏 GeoChat - RAG System 🌏            ")
    print("------------------------------------------------")
    print("Initializing Engine (This may take a moment)...")
    
    try:
        engine = RAGEngine()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Make sure you ran 'embed_chunks.py' first!")
        return

    count = engine.collection.count()
    print(f"\n🟢 GeoChat Ready — Chunks Loaded: {count}")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("------------------------------------------------")

    while True:
        try:
            query = input("\n>> ").strip()
            if not query: continue
            
            if query.lower() in ('exit', 'quit', 'q'):
                print("Goodbye! 👋")
                sys.exit(0)
                
            response = engine.answer(query)
            print("\n------------------------------------------------")
            print(response)
            print("------------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
