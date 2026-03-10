# 🌏 GeoChat - Geology RAG System

GeoChat is an advanced Retrieval-Augmented Generation (RAG) agent specialized in **Geology**. Built completely with local language models and vector databases, it accurately answers complex geological questions by retrieving contextual information from textbooks. It also features a fallback semantic router to search the internet for missing knowledge and outright refuses to answer out-of-domain (non-geology) questions to maintain system integrity.

## ✨ Features

- **End-to-End Local RAG Pipeline**: Combines text extraction, embedding, vector storage, and an LLM running locally.
- **Robust PDF Processing**: Custom pipeline to parse geology textbooks with `pdfplumber` and an OCR fallback (`pytesseract`) for scanned images or non-text-selectable pages.
- **Smart Chunking Strategy**: Parses documents intelligently into ~1000 token chunks with overlapping text to preserve context across page breaks.
- **Multiple Vector Store Implementations**: 
  - **ChromaDB**: Native persistent storage (`rag_engine.py`, `geochat.py`).
  - **FAISS**: High-performance in-memory index tracking (`rag_engine_faiss.py`, `geochat_faiss.py`).
- **Semantic Agent Router**: An intelligent agent (`georag/agent.py`) capable of classifying queries to:
  1. `retrieve_textbook`: Search the local geology vector index.
  2. `search_internet`: Use DuckDuckGo internet searches for current geology events (e.g., recent earthquakes).
  3. `refuse`: Decline queries unrelated to geology.
- **Powered by Local LLMs**: Uses `Ollama` with lightweight, capable models like `phi3:mini` to securely generate answers locally.

---

## 🏗️ Architecture Stack

* **LLM Backend**: Ollama (`phi3:mini`)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` / `all-mpnet-base-v2`
* **Vector Stores**: ChromaDB & FAISS
* **OCR / Extraction**: `pdfplumber`, `pytesseract`, `pdf2image`
* **Web Search**: DuckDuckGo Search (`duckduckgo_search`)

---

## 📂 Project Structure

```text
GEOCHAT/
│
├── geochat.py                # Main interactive chat application (ChromaDB)
├── geochat_faiss.py          # Main interactive chat application (FAISS)
├── rag_engine.py             # RAG Engine executing vector retrieval using ChromaDB
├── rag_engine_faiss.py       # RAG Engine executing vector retrieval using FAISS
│
├── process_pdfs.py           # Ingestion pipeline: extracts PDF text, performs OCR, and saves chunks
├── embed_chunks.py           # Embeds the chunked textbook data into ChromaDB
├── embed_chunks_faiss.py     # Embeds the chunked textbook data into FAISS
│
├── georag/                   # Agent logic and internal configuration
│   ├── agent.py              # Semantic router and tool executive handler
│   ├── config.py             # Core configuration (Paths, Chunking Params, Prompts)
│   ├── llm_interface.py      # Abstraction for Ollama API
│   └── retrieval.py          # Internal retrieval utility functions
│
└── geochat_data/             # Automated pipeline directories
    ├── raw_pdfs/             # Drop your geology textbooks (*.pdf) here
    ├── clean_text/           # Parsed raw text from books
    ├── chunks/               # Chunked pieces stored as JSON
    └── vector_db/            # Generated indexing databases for Vector Stores
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed, and a machine preferably with a CUDA-enabled GPU (configured via PyTorch).

You will need the following installed:
* [Ollama](https://ollama.com/)
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (Only required for image-based PDFs)
* [Poppler](https://poppler.freedesktop.org/) (Required for `pdf2image`)

### 2. Environment Setup
Install the Python dependencies:

```bash
pip install torch sentence-transformers chromadb faiss-cpu ollama pdfplumber pytesseract pdf2image Pillow duckduckgo_search
```

### 3. Start Ollama Model
Ensure Ollama is running and download the `phi3:mini` backend:
```bash
ollama run phi3:mini
```

### 4. Data Ingestion Pipeline

To populate the Knowledge Base, drop your geology PDF textbooks into the `geochat_data/raw_pdfs/` folder.

**Step 1:** Extract the text and create semantic HTML-like chunks:
```bash
python process_pdfs.py
```
*(This generates a detailed validation report at `geochat_data/verification_report.md`)*

**Step 2:** Generate embeddings and store them in the respective Vector DB:
* For ChromaDB: `python embed_chunks.py`
* For FAISS: `python embed_chunks_faiss.py`

### 5. Chat with the Geologist
Now you are ready to query the RAG system!
* Using ChromaDB: `python geochat.py`
* Using FAISS: `python geochat_faiss.py`

> **Note:** Run the interactive terminal to perform searches like "What is the geology behind the Himalayas?" or "Explain the Theory of Plate Tectonics".

---

## 🧠 Semantic Agent & Tools

GeoChat uses a smart decision engine (`georag/agent.py`) to categorize your queries. 

* **Domain Restriction**: If you ask "How to cook a steak?", the system prevents context pollution and responds: `"I am a specialized geology assistant and cannot answer questions outside this domain."`
* **Internet Augmentation**: If the knowledge is missing from local PDFs but pertains to geology (e.g. recent seismic activity), it pulls in information via **DuckDuckGo** and processes the answer natively using the LLM.

## 📝 License
This project is open-source and intended for educational AI architecture demonstrations.
