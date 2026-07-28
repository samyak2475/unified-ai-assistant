import ollama
import gradio as gr
from PIL import Image
import io

def chat_with_image(text, image=None):
    if isinstance(text, dict):
        text = text.get("text", "")
    
    prompt = text if text else "Describe this image in detail."

    if image is not None:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        messages = [{'role': 'user', 'content': prompt, 'images': [img_bytes]}]
    else:
        messages = [{'role': 'user', 'content': prompt}]
    
    response = ollama.chat(model='llava:7b', messages=messages)
    return response['message']['content']

with gr.Blocks(title="Multimodal AI Assistant") as demo:
    gr.Markdown("# 🖼️ Multimodal AI Assistant")
    gr.Markdown("Upload an image and type your question below")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        text_input = gr.Textbox(label="Your Question", placeholder="Describe this image...", lines=3)
    
    output = gr.Textbox(label="Response", lines=12)
    submit_btn = gr.Button("Submit", variant="primary")
    
    submit_btn.click(
        fn=chat_with_image,
        inputs=[text_input, image_input],
        outputs=output
    )

demo.launch()