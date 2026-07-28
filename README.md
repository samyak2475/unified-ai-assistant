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

<img width="1920" height="1080" alt="Screenshot (732)" src="https://github.com/user-attachments/assets/365cdb21-57b1-41d1-9cf5-4821985fc4a8" />
<img width="1920" height="1080" alt="Screenshot (733)" src="https://github.com/user-attachments/assets/31d8bc2a-48b2-4bfc-b8ea-15f6360411b8" />
<img width="1920" height="1080" alt="Screenshot (734)" src="https://github.com/user-attachments/assets/8ccb011c-7f24-4614-a69f-4a6cc3b7eaa5" />
