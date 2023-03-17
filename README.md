# MultiChat
MultiChat：开源多模态对话语言模型 | An open multimodal dialogue language model

## 介绍

MultiChat 是一个开源的、支持视觉（图片）的对话语言模型，具有约74亿参数，基于Blip2和ChatGLM-6B。目前MultiChat支持视觉和文本的对话，我们后续会开源支持视觉、文本、语音的多模态对话模型。

由于仅使用了4M左右的中文图文对进行模型训练，目前有较多局限性，包括MultiChat视觉感知能力及ChatGLM本身的局限性。

## 使用方式
使用 pip 安装依赖：`pip install -r requirements.txt`，其中 `transformers` 库版本推荐为 `4.26.1`，但理论上不低于 `4.23.1` 即可。

### 代码调用
```python
>>> from transformers import AutoProcessor, AutoModel
>>> processor = AutoProcessor.from_pretrained("phellonchen/multichat", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("phellonchen/multicha", trust_remote_code=True).half().cuda()
>>> response, history = model.chat(processor, None, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 MultiChat,很高兴见到你,欢迎问我任何问题。
>>> img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
>>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
>>> question = "图里有什么？"
>>> history, output = model.chat(processor, raw_image, question, history)

```

### Demo

我们提供了一个基于 [Gradio](https://gradio.app) 的网页版 Demo 、基于 [Streamlit]的网页版Demo。使用时首先需要下载本仓库：
```shell
git clone https://github.com/phellonchen/MultiChat.git
cd MultiChat

#### 网页版 Demo
![web-demo](images/web-demo.png)
首先安装 Gradio：`pip install gradio`，Streamlit: `pip install streamlit steamlit-chat`，然后运行仓库中的 [multichat_gradio.py](multichat_gradio.py)或者[multichat_streamlit.py](multichat_streamlit.py)：

```
```shell
python multichat_gradio.py
```
或者
```shell
streamlit run multichat_streamlit.py
```

程序会运行一个 Web Server，并输出地址。在浏览器中打开输出的地址即可使用。

## MultiChat 示例


## 协议

本仓库的代码依照 [Apache-2.0](LICENSE) 协议开源，并使用了 ChatGLM-6B 模型的权重，相关使用则均需要遵循 [Model License](MODEL_LICENSE)。

## 单位
中国科学院自动化研究所 听觉模型与认知计算
