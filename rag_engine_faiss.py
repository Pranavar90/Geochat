import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import torch
import os
import pickle

class RAGEngine:
    def __init__(self, db_path="geochat_data/vector_db", model_name="sentence-transformers/all-mpnet-base-v2", llm_model="phi3:mini"):
        self.db_path = db_path
        self.model_name = model_name
        self.llm_model = llm_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("  Loading Embedding Model...", end="\r")
        self.embed_model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)
        print("  Loading Embedding Model... [OK]")
        
        print("  Loading FAISS Index...", end="\r")
        self.index = faiss.read_index(os.path.join(self.db_path, "faiss.index"))
        with open(os.path.join(self.db_path, "metadata.pkl"), 'rb') as f:
            self.data = pickle.load(f)
        print(f"  Loading FAISS Index... [OK] ({self.index.ntotal} vectors)")

    def embed_query(self, query):
        return self.embed_model.encode([query]).astype('float32')

    def retrieve(self, query, k=5):
        query_embedding = self.embed_query(query)
        distances, indices = self.index.search(query_embedding, k)
        
        results = {
            'ids': [[self.data['ids'][idx] for idx in indices[0]]],
            'distances': [distances[0].tolist()],
            'metadatas': [[self.data['metadata'][idx] for idx in indices[0]]],
            'documents': [[self.data['texts'][idx] for idx in indices[0]]]
        }
        return results

    def build_prompt(self, context_list, query):
        context_str = "\n\n---\n\n".join(context_list)
        prompt = f"""You are GeoChat, a geology expert.
Use ONLY the context chunks below to answer.
If answer is not in the context, say "Not found in the provided documents."

<CONTEXT>
{context_str}
</CONTEXT>

Question: {query}
Answer with citations using the format [Book Name, Section] if available.
"""
        return prompt

    def answer(self, query, k=5):
        # 1. Retrieve
        print(f"  🔍 Retrieving top {k} chunks...", end="\r")
        retrieval_results = self.retrieve(query, k=k)
        
        if not retrieval_results['documents'][0]:
            return "No relevant documents found."

        # Extract context and metadata
        documents = retrieval_results['documents'][0]
        metadatas = retrieval_results['metadatas'][0]
        distances = retrieval_results['distances'][0]
        
        # 2. Build Prompt
        prompt = self.build_prompt(documents, query)
        
        # 3. Call LLM
        print("  🤖 Generating answer...        ", end="\r")
        try:
            response = ollama.chat(model=self.llm_model, messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            answer_text = response['message']['content']
            
            # Format Citations
            citations = []
            for meta, dist in zip(metadatas, distances):
                citations.append(f"- {meta.get('book', 'Unknown')} (Dist: {dist:.4f})")
            
            final_output = f"{answer_text}\n\n**Sources:**\n" + "\n".join(citations[:3])
            return final_output
            
        except Exception as e:
            return f"Error calling Ollama: {e}\nEnsure 'ollama serve' is running and you have pulled '{self.llm_model}'."

if __name__ == "__main__":
    # Interactive Test
    engine = RAGEngine()
    print("\n" + "="*50)
    print("RAG Engine Interactive Test")
    print("Type 'exit', 'quit', or 'q' to stop")
    print("="*50)
    
    while True:
        query = input("\n>> ").strip()
        if not query:
            continue
        if query.lower() in ('exit', 'quit', 'q'):
            print("Goodbye!")
            break
        
        print("\n" + "-"*50)
        print(engine.answer(query))
        print("-"*50)
