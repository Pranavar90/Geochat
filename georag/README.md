# 🌏 GeoRAG - Advanced Geology QA System

A local, high-performance RAG pipeline for answering geology questions using Ollama and FAISS.

## Features
- **Local & Offline**: Uses Ollama (Phi-3) and local embeddings (MPNet).
- **Fast Retrieval**: FAISS vector database.
- **GPU Accelerated**: Faster ingestion and inference.
- **Modular**: Clean project structure.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Ollama Model**
   Ensure Ollama is running (`ollama serve`).
   ```bash
   ollama pull phi3:mini
   ```

3. **Prepare Data**
   Place your PDF textbooks in `georag/data/`.

## Usage

1. **Ingest Data** (Run this first or when adding new PDFs)
   ```bash
   python -m georag.ingest
   ```

2. **Start Chat**
   ```bash
   python -m georag.main
   ```

## Structure
- `georag/config.py`: Configuration settings.
- `georag/ingest.py`: PDF processing and vector database creation.
- `georag/main.py`: Interactive CLI.
