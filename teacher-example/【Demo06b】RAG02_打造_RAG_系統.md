# 【Demo06b】RAG02_打造_RAG_系統

*This notebook was created for Google Colab*

*Language: python*

---


### 0. 讀入你打造好的 vector dataset



```python
!pip -q install gdown
```



```python
GDRIVE_PUBLIC_URL = "https://drive.google.com/file/d/1raCLp-CkB4scczsHU-BWwG1Tv4-9pWOU/view?usp=sharing"
```



```python
!gdown --fuzzy -O faiss_db.zip "{GDRIVE_PUBLIC_URL}"
```



```python
!unzip faiss_db.zip
```


### 1. 安裝並引入必要套件



```python
!pip install -U langchain langchain-community faiss-cpu transformers sentence-transformers huggingface_hub
!pip -q install "aisuite[all]"
```



```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
```



```python
from openai import OpenAI
import gradio as gr
```


### 2. 自訂 E5 embedding 類別



```python
import os
from google.colab import userdata
```



```python
hf_token = userdata.get('HuggingFace')
```



```python
from huggingface_hub import login
login(token=hf_token)
```



```python
class EmbeddingGemmaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="google/embeddinggemma-300m",
            encode_kwargs={"normalize_embeddings": True},
            **kwargs
        )

    def embed_documents(self, texts):
        # 你也可以把 "none" 改成真實標題（檔名/章節名），效果會更穩
        texts = [f"title: none | text: {t}" for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        # 官方檢索建議前綴
        return super().embed_query(f"task: search result | query: {text}")
```


### 3. 載入 `faiss_db`



```python
embedding_model = EmbeddingGemmaEmbeddings()
vectorstore = FAISS.load_local(
    "faiss_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
```



```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```


### 4. 設定好我們要的 LLM


如之前, 我們會用 OpenAI API。這裡使用 Groq 服務, 可改成你要的服務。



```python
import aisuite as ai
```



```python
api_key = userdata.get('Groq')
```



```python
os.environ['GROQ_API_KEY']=api_key
```


這裡的模型和 `base_url` 是用 Groq, 如果用其他服務請自行修改。



```python
model = "groq:openai/gpt-oss-120b"
#base_url="https://api.groq.com/openai/v1"
```



```python
client = ai.Client()
```


### 5. `prompt` 設計



```python
system_prompt = "你是政大的 AI 自主學習輔導員，請根據資料來回應學生的問題。請親切、簡潔並附帶具體建議。請用台灣習慣的中文回應。"

prompt_template = """
根據下列資料：
{retrieved_chunks}

回答使用者的問題：{question}

請根據資料內容回覆，若資料不足請告訴同學可以請教學務處生僑組的老師。
"""
```


### 6. 使用 RAG 來回應

搜尋與使用者問題相關的資訊，根據我們的 prompt 樣版去讓 LLM 回應。



```python
chat_history = []

def chat_with_rag(user_input):
    global chat_history
    # 取回相關資料
    docs = retriever.get_relevant_documents(user_input)
    retrieved_chunks = "\n\n".join([doc.page_content for doc in docs])

    # 將自定 prompt 套入格式
    final_prompt = prompt_template.format(retrieved_chunks=retrieved_chunks, question=user_input)

    # 用 AI Suite 呼叫語言模型
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_prompt},
    ]
    )
    answer = response.choices[0].message.content

    chat_history.append((user_input, answer))
    return answer
```


### 7. 用 Gradio 打造 Web App



```python
with gr.Blocks() as demo:
    gr.Markdown("# 🎓 AI 學校獎懲諮詢師")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="請輸入你的問題...")

    def respond(message, chat_history_local):
        response = chat_with_rag(message)
        chat_history_local.append((message, response))
        return "", chat_history_local

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(debug=True)
```
