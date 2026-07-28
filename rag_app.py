import ollama
import gradio as gr
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None

def process_pdfs(files):
    global vectorstore
    if not files:
        return "Please upload PDF(s)."
    
    # Clear everything old
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    all_docs = []
    for file in files:
        loader = PyPDFLoader(file.name)
        docs = loader.load()
        all_docs.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
    
    return f"✅ Loaded {len(files)} PDF(s) — Ready to answer questions!"

def rag_chat(question):
    global vectorstore
    if vectorstore is None:
        return "Please upload PDF(s) first."
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    context_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content[:1200] for doc in context_docs])
    
    prompt = f"""Answer the question **only** using the context from the currently uploaded document. Do not use external knowledge.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(model='phi3:mini', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']


with gr.Blocks(title="Smart Document Assistant") as demo:
    gr.Markdown("# 📚 Smart Document Assistant")
    gr.Markdown("**Upload PDF(s) → Ask questions (Only current upload is used)**")
    
    with gr.Tab("Upload PDF"):
        file_upload = gr.File(file_count="multiple", file_types=[".pdf"], label="Upload your PDF(s)")
        upload_btn = gr.Button("Process PDFs", variant="primary")
        status = gr.Textbox(label="Status")
        upload_btn.click(process_pdfs, inputs=file_upload, outputs=status)
    
    with gr.Tab("Ask Questions"):
        question_input = gr.Textbox(lines=3, placeholder="Summarize this document...", label="Your Question")
        ask_btn = gr.Button("Get Answer", variant="primary")
        answer_output = gr.Textbox(lines=20, label="Answer")
        
        ask_btn.click(rag_chat, inputs=question_input, outputs=answer_output)

demo.launch()



