import gradio as gr
import os
import uuid
from PIL import Image
import numpy as np
import re
from transformers import AutoModel, AutoProcessor


processor = AutoProcessor.from_pretrained("phellonchen/multichat", trust_remote_code=True)
model = AutoModel.from_pretrained("phellonchen/multichat", trust_remote_code=True).hafl().cuda()
model = model.eval()

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2
Current_Image = None

def predict(input, history=[]):
    global Current_Image
    # response, history = model.chat(tokenizer, input, history)
    if Current_Image is not None:
        raw_image = Image.open(Current_Image.name).convert('RGB')
    else:
        raw_image = None
    history, output = model.chat(processor, raw_image, input, history=history)
    
    return history, history

def predict_image(image, state=[], input=None):
    global Current_Image
    Current_Image = image
    AI_prompt = "收到。"
    state = state + [((image.name,), AI_prompt)]

    return state, state

if __name__ == '__main__':
    with gr.Blocks(css="#chatbot{height:520px; overflow:auto;} .overflow-y-auto") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="MultiChat")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["image"])

        txt.submit(predict, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(predict_image, [btn, state, txt], [chatbot, state])
        clear.click(lambda: [], None, Current_Image)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)

        demo.queue().launch(share=True, inbrowser=True)
