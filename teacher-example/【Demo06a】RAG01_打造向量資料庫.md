# 【Demo06a】RAG01_打造向量資料庫

*This notebook was created for Google Colab*

*Language: python*

---


暫時沒有任何資料，可以找一個學校你有興趣的規定，比如[政大學生獎懲辦法](https://osa.nccu.edu.tw/files/1431422025610b81850375e.pdf)來練習。


### 1. 建立資料夾



```python
import os
upload_dir = "uploaded_docs"
os.makedirs(upload_dir, exist_ok=True)
print(f"請將你的 .txt, .pdf, .docx 檔案放到這個資料夾中： {upload_dir}")
```


<font color="red">請手動上傳自己的檔案再繼續。</font>


### 2. 更新必要套件並引入



```python
!pip install -U langchain langchain-community pypdf python-docx faiss-cpu
!pip install -U sentence-transformers transformers
```



```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
```


### 3. 依 Google 建議加入 EmbeddingGemma 前綴

Google 建議, 在文本部份的 Embedding 要用以下格式:

    title: {title|none} | text: ...

而問題 Query 要用:

    Query：task: search result | query: ...



```python
from langchain.embeddings import HuggingFaceEmbeddings
```



```python
class EmbeddingGemmaEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="google/embeddinggemma-300m",   # HF 上的官方模型
            encode_kwargs={"normalize_embeddings": True},  # 一般檢索慣例
            **kwargs
        )

    def embed_documents(self, texts):
        # 文件向量：title 可用 "none"，或自行帶入檔名/章節標題以微幅加分
        texts = [f'title: none | text: {t}' for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        # 查詢向量：官方建議的 Retrieval-Query 前綴
        return super().embed_query(f'task: search result | query: {text}')
```


### 4. 載入文件



```python
folder_path = upload_dir
documents = []
for file in os.listdir(folder_path):
    path = os.path.join(folder_path, file)
    if file.endswith(".txt"):
        loader = TextLoader(path)
    elif file.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif file.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(path)
    else:
        continue
    documents.extend(loader.load())
```


### 5. 建立向量資料庫



```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(documents)
```



```python
from google.colab import userdata
hf_token = userdata.get('HuggingFace')
```



```python
from huggingface_hub import login
login(token=hf_token)
```



```python
embedding_model = EmbeddingGemmaEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding_model)
```


### 6. 儲存向量資料庫



```python
vectorstore.save_local("faiss_db")
```



```python
!zip -r faiss_db.zip faiss_db
```



```python
print("✅ 壓縮好的向量資料庫已儲存為 'faiss_db.zip'，請下載此檔案備份。")
```
