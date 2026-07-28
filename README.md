# 🤖 Unified AI Assistant

**Multimodal + RAG AI System** — General Chat, Image Understanding & Document Q&A

A complete local AI application combining multiple capabilities in one interface.

## Features
- **General Chat**: Talk with Phi-3 mini
- **Image Analysis**: Upload image + ask questions (LLaVA)
- **Document RAG**: Upload PDFs → Ask intelligent questions about them
- All running **locally** on laptop (no internet needed after setup)

## Tech Stack
- **LLMs**: Phi-3 Mini + LLaVA (via Ollama)
- **RAG**: LangChain + ChromaDB + HuggingFace Embeddings
- **UI**: Gradio
- **Document Processing**: PyPDFLoader

## How to Run
1. Install Ollama + models:
   ```bash
   ollama pull phi3:mini
   ollama pull llava:7b

2. Install dependencies:    
    pip install gradio ollama langchain langchain-community langchain-text-splitters pypdf chromadb sentence-transformers pillow

3. run:
    python final_app.py

What I Learned

Building multimodal applications
Implementing RAG pipelines from scratch
Working with local LLMs and vector databases
Creating clean multi-tab Gradio interfaces
Optimizing for low-resource hardware (4GB GPU)

Project demonstrates end-to-end AI system development skills.