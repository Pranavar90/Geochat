import sys
import time
from . import config, retrieval, llm_interface, utils

def format_prompt(query, context_results):
    """Builds the prompt for the LLM."""
    context_str = ""
    sources = []
    
    for i, res in enumerate(context_results):
        context_str += f"\n[Chunk {i+1}]\n{res['text']}\n"
        meta = res['metadata']
        sources.append(f"{meta['source']} (Page {meta['page']})")
        
    prompt = f"""You are an expert geology tutor. Use ONLY the retrieved textbook context below to answer.
If the answer is not in the context, say "I cannot find the answer in the provided text."

<CONTEXT>
{context_str}
</CONTEXT>

QUESTION: {query}

Answer with specific references to the provided chunks.
"""
    return prompt, list(set(sources))

def main():
    print("="*60)
    print("              🌏 GeoRAG - Advanced Geology QA 🌏")
    print("             Local RAG System (Ollama + FAISS)")
    print("="*60)
    
    # Initialize
    try:
        retriever = retrieval.Retriever()
        llm = llm_interface.LLMInterface()
    except Exception as e:
        print(f"\n❌ Initialization Failed: {e}")
        print("Tip: Run 'python -m georag.ingest' first!")
        return

    print("\n✅ System Ready. Type 'exit' to quit.\n")
    
    while True:
        try:
            query = input("\n📝 Question >> ").strip()
            if not query: continue
            if query.lower() in ('exit', 'quit', 'q'):
                print("Goodbye!")
                break
            
            # 1. Retrieve
            print("  🔍 Searching knowledge base...", end="\r")
            start_time = time.time()
            results = retriever.retrieve(query)
            if not results:
                print("\n⚠️ No relevant information found.")
                continue
                
            # 2. Build Prompt
            prompt, sources = format_prompt(query, results)
            
            # 3. Generate
            print("  🤖 Generating answer...       ", end="\r")
            
            print("\n" + "-"*60)
            response_stream = llm.generate_response(prompt, stream=True)
            
            full_answer = ""
            for chunk in response_stream:
                content = chunk['message']['content']
                print(content, end="", flush=True)
                full_answer += content
                
            print("\n" + "-"*60)
            print(f"\n📚 Sources:")
            for s in sources:
                print(f" - {s}")
                
            elapsed = time.time() - start_time
            print(f"\n⏱️ Time: {elapsed:.2f}s")
            
        except KeyboardInterrupt:
            print("\nAborted.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
