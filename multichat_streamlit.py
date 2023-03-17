import streamlit as st
from PIL import Image
import torch
import requests
from PIL import Image
from transformers import AutoModel, AutoProcessor



import numpy as np
import cv2
from streamlit_chat import message

# Streamlit App
st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
device = 'cpu'
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("/raid/cfl/pretraining/en/demo/multichat/checkpoints/multichat", trust_remote_code=True)
    # use gpu
    # model = AutoModel.from_pretrained("/raid/cfl/pretraining/en/demo/multichat/checkpoints/multichat", trust_remote_code=True).half().to(device)
    # use cpu
    model = AutoModel.from_pretrained("/raid/cfl/pretraining/en/demo/multichat/checkpoints/multichat", trust_remote_code=True).float().to(device)
    return model, processor

def MulitChatgpt(image, query, history, model, processor):
    history, output = model.chat(processor, image, query, history=history)
    return history, output

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'use_image' not in st.session_state:
    st.session_state['use_image'] = False

def get_image():
    uploaded_file = st.file_uploader("You can upload an image to talk with me")
    if uploaded_file is not None:
    # 将传入的文件转为Opencv格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # 展示图片
        st.image(opencv_image, channels="BGR")
        # 保存图片
        # cv2.imwrite('test.jpg',opencv_image)
        st.session_state['use_image'] = True
        return Image.open(uploaded_file).convert("RGB")
    else:
        st.session_state['use_image'] = False
        return None
    

def get_text():
    # state = st.selectbox('Model Prompt',('Chat', 'TextQA', 'Doc-based Dialog', "Image Chat", "VQA", "Visual Dialog", "Speech Dialog", "Visual Speech Dialog","ASR"))
    state = None
    queryText = st.text_area("Message:", value="Let's talk about something!", height=3, max_chars=None ) 
   
    tab1, tab2 = st.tabs(["Send", "Clear"])
    # tab1, tab2, tab3 = st.tabs(["Send", "Clear", "Download"])
    with tab1:
        btnResult = st.button('Send Message')
    with tab2: 
        clrResult= st.button('Clear History') 
    # with tab3:
    #     text_contents = ''
    #     for i in range(len(st.session_state['generated'])):
    #             text_contents += "User Turn %d: " % (i+1) +  st.session_state['past'][i] + '\n'
    #             text_contents += "Bot Turn %d: " % (i+1) + st.session_state["generated"][i]  + '\n'
    #     st.download_button('Download Dialog', text_contents)

    if clrResult:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['history'] = []
        st.session_state['use_image'] = False
    if queryText:                     
        if btnResult:
            return queryText, state

    return False, state

def main(model, processors, device):
    image = get_image()

    col1, col2 = st.columns([5, 2])

    with col2:
        user_input, model_state = get_text()
    
    with col1:
        if user_input:
            history_input = st.session_state.history
            output = MulitChatgpt(image, user_input, history_input, model, processors)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output[1])
            # st.session_state.history.append((user_input, output[1]))

        if st.session_state['generated']:
            order = True
            if not order:   
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message("Turn %d: "% (i+1)+ st.session_state["generated"][i], key=str(i))
                    message("Turn %d: "% (i+1)+ st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    # message(st.session_state["generated"][i], key=str(i))
            else:
                for i in range(len(st.session_state['generated'])):
                    # message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))
        else:
        
            st.session_state['generated'].append("让我们开始聊天吧！")
            # st.session_state['past'].append("Hi, i'm glad to talk with you!")
            message(st.session_state['generated'][0], key=str(-1))
            st.session_state['generated'] = []

        hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


if __name__ == '__main__':
    model, processors = load_model()
    st.header("MultiChat with Streamlit")
    main(model, processors, device)
    

    