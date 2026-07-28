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

# UI
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





# import ollama
# import gradio as gr
# import os
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# def load_documents(folder_path="documents"):
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#         print(f"✅ Created '{folder_path}' folder. Add your PDF files there.")
#         return []
    
#     loader = PyPDFDirectoryLoader(folder_path)
#     docs = loader.load()
#     print(f"✅ Loaded {len(docs)} documents from PDFs")
    
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     split_docs = text_splitter.split_documents(docs)
#     print(f"✅ Split into {len(split_docs)} chunks")
#     return split_docs

# # Load documents and create vector store
# docs = load_documents()

# if docs:
#     vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
#     print("✅ Vector store created successfully")
# else:
#     vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# def rag_chat(question):
#     if not question or not question.strip():
#         return "Please ask a question."
    
#     # Updated method
#     context_docs = retriever.invoke(question)
#     context = "\n\n".join([doc.page_content for doc in context_docs])
    
#     prompt = f"""Use the following context to answer the question accurately.
# If you don't know the answer, just say "I don't know based on the documents."

# Context:
# {context}

# Question: {question}

# Answer:"""

#     response = ollama.chat(
#         model='phi3:mini',
#         messages=[{'role': 'user', 'content': prompt}]
#     )
#     return response['message']['content']

# # Gradio UI
# with gr.Blocks(title="Smart Document Assistant") as demo:
#     gr.Markdown("# 📚 Smart Document Assistant (RAG)")
#     gr.Markdown("**Upload PDFs in the 'documents' folder and ask questions**")
    
#     with gr.Row():
#         question_input = gr.Textbox(
#             label="Your Question", 
#             lines=3, 
#             placeholder="What is this document about? Summarize..."
#         )
#         submit_btn = gr.Button("Ask", variant="primary")
    
#     answer_output = gr.Textbox(label="Answer", lines=12)
    
#     submit_btn.click(
#         fn=rag_chat,
#         inputs=question_input,
#         outputs=answer_output
#     )

# demo.launch()