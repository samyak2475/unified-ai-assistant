import ollama
import gradio as gr
import os
import shutil
from PIL import Image
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ===================== SETUP =====================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None

# ===================== FUNCTIONS =====================
def process_pdfs(files):
    global vectorstore
    if not files:
        return "Please upload PDF(s)."
    
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    all_docs = []
    for file in files:
        loader = PyPDFLoader(file.name)
        all_docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
    return f"✅ Processed {len(files)} PDF(s) — Ready!"

def rag_chat(question):
    global vectorstore
    if vectorstore is None:
        return "Please upload PDFs first in Document tab."
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    context_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content[:1000] for doc in context_docs])
    
    prompt = f"Answer based only on context:\nContext: {context}\nQuestion: {question}\nAnswer:"
    response = ollama.chat(model='phi3:mini', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def image_chat(message, image):
    if image is None:
        return "Please upload an image."
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    messages = [{'role': 'user', 'content': message or "Describe this image.", 'images': [img_bytes]}]
    response = ollama.chat(model='llava:7b', messages=messages)
    return response['message']['content']

def general_chat(message, history):
    response = ollama.chat(model='phi3:mini', messages=[{'role': 'user', 'content': message}])
    return response['message']['content']

# ===================== MAIN UI =====================
with gr.Blocks(title="Unified AI Assistant") as demo:
    gr.Markdown("# 🤖 Unified AI Assistant")
    gr.Markdown("**General Chat + Image Understanding + Document RAG** — All in one app")

    with gr.Tab("💬 General Chat"):
        gr.ChatInterface(general_chat, title="General Chat with Phi-3")

    with gr.Tab("🖼️ Image Analysis"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            text_input = gr.Textbox(label="Ask about the image", placeholder="What do you see? Describe it...")
        image_output = gr.Textbox(label="Response")
        gr.Button("Analyze Image", variant="primary").click(
            image_chat, inputs=[text_input, image_input], outputs=image_output
        )

    with gr.Tab("📚 Document Q&A"):
        with gr.Row():
            file_upload = gr.File(file_count="multiple", file_types=[".pdf"], label="Upload PDFs")
            upload_btn = gr.Button("Process PDFs", variant="primary")
        status = gr.Textbox(label="Upload Status")
        upload_btn.click(process_pdfs, inputs=file_upload, outputs=status)
        
        with gr.Row():
            doc_question = gr.Textbox(lines=3, placeholder="Ask anything about the uploaded documents...")
            ask_btn = gr.Button("Get Answer", variant="primary")
        doc_answer = gr.Textbox(lines=15, label="Answer")
        ask_btn.click(rag_chat, inputs=doc_question, outputs=doc_answer)

demo.launch()