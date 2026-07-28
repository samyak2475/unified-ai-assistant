import ollama
import gradio as gr

def chat(message, history):
    response = ollama.chat(
        model='phi3:mini',
        messages=[{'role': 'user', 'content': message}]
    )
    return response['message']['content']

demo = gr.ChatInterface(
    chat,
    title="My First AI Assistant",
    description="Running locally on your laptop using Phi-3 mini"
)

demo.launch()