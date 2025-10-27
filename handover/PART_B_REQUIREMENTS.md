# Part B 需求規格書

## 專案目標

建立 RAG 強化的公司客服對話機器人，結合：
- FAISS 向量資料庫（Part A 已完成）
- Agentic search（AI 可以列出檔案、讀取特定檔案）
- GPT-5 生成回答
- Gradio Web UI

---

## 使用者意圖

### 核心需求

1. **RAG 對話機器人**
   - 從 Google Drive 下載 `faiss_db.zip`
   - 載入 FAISS 向量資料庫
   - 使用者問問題 → 檢索相關 chunks → GPT-5 生成回答

2. **Agentic Search（進階功能）**
   - AI 可以「列出所有可用的 Q&A 檔案」（list_files tool）
   - AI 可以「讀取特定檔案的內容」（read_file tool）
   - AI 會先搜尋檔案清單，決定要從哪些檔案檢索答案
   - 展示「AI 在思考要去哪裡找資料」的過程

3. **人設與風格**
   - 角色：公司客服助手
   - 語氣：親切、專業、有幫助
   - 回答風格：簡潔明瞭，提供具體步驟
   - 若資料不足：建議聯繫真人客服

---

## 技術架構

### 檔案結構（預期）

```
Week7/
├── rag_chatbot.py              # Part B 主程式
├── faiss_db.zip                # Part A 產出（或從 Drive 下載）
├── faiss_db/                   # 解壓縮後的資料庫
│   ├── index.faiss
│   └── metadata.json
└── gradio_app.py               # （可選）獨立的 Gradio UI
```

### 核心功能模組

#### 1. 環境偵測與初始化

```python
import os

def is_colab():
    """偵測是否在 Colab 環境"""
    try:
        import google.colab
        return True
    except:
        return False

def setup_environment():
    """設定 API key 和環境"""
    if is_colab():
        from google.colab import userdata
        api_key = userdata.get('OPENAI_API_KEY')
    else:
        api_key = os.getenv('OPENAI_API_KEY')  # 本地從 fish env

    return api_key
```

#### 2. 下載與載入向量資料庫

```python
def download_faiss_db(gdrive_url):
    """從 Google Drive 下載 faiss_db.zip"""
    if is_colab():
        # Colab: 使用 gdown
        !gdown --fuzzy -O faiss_db.zip "{gdrive_url}"
        !unzip -o faiss_db.zip
    else:
        # 本地：假設已經存在
        if not os.path.exists("faiss_db.zip"):
            print("本地模式：請確保 faiss_db.zip 存在")

def load_vectorstore():
    """載入 FAISS 向量資料庫"""
    import faiss
    import json

    # 載入 FAISS index
    index = faiss.read_index("faiss_db/index.faiss")

    # 載入 metadata
    with open("faiss_db/metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return index, metadata
```

#### 3. RAG 檢索系統

```python
def search_similar_chunks(query, index, metadata, embedding_model, k=5):
    """
    搜尋與問題最相似的 chunks

    Args:
        query: 使用者問題
        index: FAISS index
        metadata: chunk metadata
        embedding_model: OpenAI embedding model
        k: 回傳前 k 個結果

    Returns:
        List[Dict]: 相關的 chunks
    """
    # 1. 將問題轉成 embedding
    from openai import OpenAI
    client = OpenAI()

    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[query]
    )
    query_embedding = response.data[0].embedding

    # 2. FAISS 搜尋
    import numpy as np
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, k)

    # 3. 取得相關 chunks
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])

    return results
```

#### 4. Agentic Search Tools（進階功能）

```python
def list_available_files(metadata):
    """列出所有可用的來源檔案"""
    files = set(chunk["source_file"] for chunk in metadata)
    return list(files)

def read_file_chunks(filename, metadata):
    """讀取特定檔案的所有 chunks"""
    file_chunks = [
        chunk for chunk in metadata
        if chunk["source_file"] == filename
    ]
    return file_chunks

# 定義 GPT-5 tools
AGENTIC_TOOLS = [
    {
        "type": "function",
        "name": "list_files",
        "description": "列出所有可用的 Q&A 來源檔案。當你需要了解有哪些資料可以查詢時使用。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "讀取特定檔案的完整內容。當你需要深入了解某個檔案的資訊時使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "要讀取的檔案名稱，例如 'function calling.md'"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "type": "function",
        "name": "search_chunks",
        "description": "在向量資料庫中搜尋與問題相關的內容片段。這是主要的搜尋功能。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "要搜尋的問題或關鍵字"
                },
                "k": {
                    "type": "number",
                    "description": "回傳前 k 個最相關的結果，預設 5"
                }
            },
            "required": ["query"]
        }
    }
]
```

#### 5. GPT-5 對話生成

```python
def generate_response(user_question, retrieved_chunks, chat_history=[]):
    """
    使用 GPT-5 生成回答

    Args:
        user_question: 使用者問題
        retrieved_chunks: 檢索到的相關 chunks
        chat_history: 對話歷史（可選）

    Returns:
        str: AI 回答
    """
    from openai import OpenAI
    client = OpenAI()

    # 組合 context
    context = "\n\n".join([
        f"[來源：{chunk['source_file']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    # Prompt 設計
    system_prompt = """你是一位專業的公司客服助手。
請根據提供的資料來回答客戶的問題。

回答風格：
- 親切、專業、有幫助
- 簡潔明瞭，提供具體步驟
- 如果資料中沒有答案，誠實告知並建議聯繫真人客服

請用繁體中文（台灣）回答。"""

    user_prompt = f"""參考資料：
{context}

客戶問題：{user_question}

請根據參考資料回答客戶的問題。"""

    # 呼叫 GPT-5
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "low"},  # 客服回答需要一定推理能力
        instructions=system_prompt,
        input=[{"role": "user", "content": user_prompt}]
    )

    return response.output_text
```

#### 6. Gradio UI

```python
import gradio as gr

def chatbot_interface(message, history):
    """
    Gradio chatbot 介面

    Args:
        message: 使用者輸入
        history: 對話歷史

    Returns:
        str: AI 回答
    """
    # 1. 檢索相關 chunks
    chunks = search_similar_chunks(message, index, metadata, k=5)

    # 2. 生成回答
    response = generate_response(message, chunks, history)

    return response

# 建立 Gradio 介面
with gr.Blocks(title="公司客服 AI 助手") as demo:
    gr.Markdown("# 🤖 公司客服 AI 助手")
    gr.Markdown("我可以幫您解答關於公司產品、服務、政策的問題。")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(
        placeholder="請輸入您的問題...",
        label="您的問題"
    )
    clear = gr.Button("清除對話")

    msg.submit(chatbot_interface, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True, debug=True)
```

---

## Agentic Search 工作流程

### 使用者問：「如何申請退款？」

**標準 RAG 流程：**
1. 將問題轉成 embedding
2. FAISS 搜尋最相關的 5 個 chunks
3. 把 chunks 丟給 GPT-5 生成回答

**Agentic Search 流程：**
1. GPT-5 收到問題
2. GPT-5 決定：「我需要先看看有哪些資料」→ 呼叫 `list_files`
3. 系統回傳：`["company_faq.md", "refund_policy.md", ...]`
4. GPT-5 決定：「退款問題應該在 refund_policy.md」→ 呼叫 `read_file("refund_policy.md")`
5. 系統回傳該檔案的所有 chunks
6. GPT-5 從這些 chunks 中找到答案
7. 生成回答

**優勢：**
- 使用者看到 AI 的思考過程
- 更精準地定位資料來源
- 展示 AI 的主動性

---

## 作業繳交要求

### Colab Notebook 結構

```python
# ====== Part A: 建立向量資料庫 ======
# (展示 build_vectordb.py 的程式碼和說明)
# 註明：「此部分已在本地執行完成」

# ====== Part B: RAG 對話機器人 ======

## 1. 環境設定
# 安裝套件
!pip install openai faiss-cpu numpy gradio

## 2. 下載向量資料庫
GDRIVE_URL = "https://drive.google.com/file/d/YOUR_ID/view?usp=sharing"
# 下載並解壓縮

## 3. 載入向量資料庫
# 載入 FAISS index 和 metadata

## 4. RAG 系統
# 實作檢索和生成功能

## 5. Gradio 介面
# 啟動對話機器人
demo.launch(share=True)
```

### 截圖內容

1. **資料說明**：展示使用的 Q&A 資料（qa_data/ 檔案清單）
2. **人設設定**：展示 system prompt 和角色設定
3. **Gradio 對話結果**：
   - 至少 3 個問答範例
   - 展示回答品質
   - （如果有 agentic search）展示 AI 思考過程

### Markdown 說明

```markdown
# Week 7 作業：公司客服 RAG 對話機器人

## 資料來源
我使用的資料是...（說明你的 Q&A 資料）

## 人設設定
角色：專業的公司客服助手
語氣：親切、有幫助
特色：...

## 技術架構
- Part A: GPT-5 智能分塊 + FAISS 向量資料庫
- Part B: RAG 檢索 + GPT-5 生成 + Gradio UI
- （可選）Agentic Search: AI 可以列出檔案、讀取檔案

## 對話範例
（貼截圖）
```

---

## 評分重點

根據作業要求：
- **6分**：主題與老師範例相似（校園規定查詢）
- **7-9分**：有趣的主題 + 完整功能
- **10分**：完美！有創意、功能完整、展示良好

### 加分項目
- ✅ 主題有趣（公司客服比校規有趣）
- ✅ Agentic search 功能（展示 AI 思考過程）
- ✅ 對話品質高（回答準確、語氣自然）
- ✅ UI 美觀（Gradio 客製化）
- ✅ 說明清楚（Markdown + 截圖）

---

## 開發建議

### 優先順序

**第一階段（基本功能）：**
1. 載入 FAISS 向量資料庫
2. 實作基本 RAG（檢索 + 生成）
3. Gradio 基本介面
4. 測試 3-5 個問答

**第二階段（進階功能）：**
1. 加入 agentic search tools
2. 展示 AI 思考過程
3. 美化 Gradio UI
4. 準備截圖和說明

### 測試建議

**本地測試：**
```bash
cd Week7
uv run python rag_chatbot.py
```

**Colab 測試：**
- 確保 Google Drive 連結可下載
- 測試 API key 讀取
- 測試 Gradio share link

---

## 參考資料

### 老師範例

1. **Demo06a**：建立向量資料庫（已完成 Part A）
2. **Demo06b**：RAG 系統
   - 使用 LangChain
   - Groq API
   - 政大規定查詢

3. **Demo06**：心靈處方籤機器人
   - 隨機抽籤功能
   - 結合書籍內容
   - 溫和的回答風格

### 技術文件

- OpenAI Response API：`response API OpenAI GPT 5/`
- GPT-5 tool calling：`function calling.md`
- Text generation：`text generation.md`

---

## 重要提醒

1. **必須包含老師的固定套件**
   - LangChain（或類似的 RAG 框架）
   - FAISS
   - Gradio
   - 缺一扣 1 分！

2. **Colab 權限**
   - 「知道連結的任何人」可檢視
   - Google Drive 連結要能下載

3. **AI 協助說明**
   - 如果用 AI 幫忙，要說明哪裡用了
   - 附上 Prompt 和結果截圖
   - 加上自己的理解說明

4. **截止時間**
   - 10/27 23:59（今天！）
   - 遲交期限：10/28 23:59

---

## 下一步行動

1. ✅ 閱讀本文件
2. ⬜ 實作 `rag_chatbot.py`
3. ⬜ 測試本地執行
4. ⬜ 準備 Colab notebook
5. ⬜ 上傳 faiss_db.zip 到 Google Drive
6. ⬜ 測試 Colab 執行
7. ⬜ 截圖和說明
8. ⬜ 繳交作業

**祝開發順利！🚀**
