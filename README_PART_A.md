# Part A: 建立向量資料庫

## 功能說明

使用 GPT-5 進行智能語義分塊，結合 OpenAI Embeddings 和 FAISS 建立向量資料庫。

### 核心特性

- ✅ **並行處理**：多個檔案同時處理（GPT-5 + embeddings）
- ✅ **智能分塊**：GPT-5 理解 Q&A 結構，語義完整
- ✅ **即時進度**：顯示處理進度（字數百分比）
- ✅ **自動重試**：Embedding API 失敗自動重試 3 次
- ✅ **可追蹤**：所有 chunks 保存成 JSON

## 安裝依賴

使用 `uv` 管理套件：

```bash
# 初始化專案（只需執行一次）
uv init

# 安裝依賴
uv add openai faiss-cpu numpy
```

## 使用方式

### 1. 準備資料

將你的 Q&A Markdown 檔案放到 `qa_data/` 目錄：

```bash
qa_data/
├── company_faq_1.md
├── company_faq_2.md
└── ...
```

### 2. 設定 API Key

確保你的環境變數有 `OPENAI_API_KEY`：

```bash
# Fish shell (自動載入)
export OPENAI_API_KEY="sk-..."

# 或在 ~/.config/fish/config.fish 設定
set -x OPENAI_API_KEY "sk-..."
```

### 3. 執行程式

```bash
uv run python build_vectordb.py
```

### 4. 輸出

程式會產生：

```
chunks/                    # GPT-5 分塊結果（JSON）
├── company_faq_1.json
├── company_faq_2.json
└── ...

faiss_db/                  # 向量資料庫
├── index.faiss           # FAISS 索引
└── metadata.json         # Chunk 元數據

faiss_db.zip              # 最終打包檔案
```

### 5. 上傳到 Google Drive

手動上傳 `faiss_db.zip` 到 Google Drive：
1. 上傳檔案
2. 設定為「知道連結的任何人」可檢視
3. 取得分享連結（格式：`https://drive.google.com/file/d/{ID}/view?usp=sharing`）

## 輸出格式

### chunks/{filename}.json

```json
[
  {
    "chunk_id": "call_abc123",
    "text": "Q: 如何申請退款？\nA: 您可以透過以下步驟...",
    "source_file": "company_faq_1.md",
    "chunk_index": 0,
    "char_count": 128
  }
]
```

### faiss_db/metadata.json

```json
[
  {
    "chunk_id": "call_abc123",
    "text": "Q: 如何申請退款？\nA: ...",
    "source_file": "company_faq_1.md",
    "chunk_index": 0,
    "vector_index": 0,
    "embedding_model": "text-embedding-3-large",
    "char_count": 128
  }
]
```

## 設定調整

可在 `build_vectordb.py` 頂部修改：

```python
EMBEDDING_MODEL = "text-embedding-3-large"  # Embedding 模型
EMBEDDING_BATCH_SIZE = 100                  # 批次大小
MAX_WORKERS = 5                             # 並行處理數量
RETRY_ATTEMPTS = 3                          # 重試次數
```

## 故障排除

### 錯誤：找不到 qa_data 目錄
```bash
mkdir qa_data
# 然後放入 .md 檔案
```

### 錯誤：OPENAI_API_KEY not found
```bash
# 設定環境變數
export OPENAI_API_KEY="sk-..."
```

### GPT-5 分塊失敗
- 檢查檔案編碼是否為 UTF-8
- 檢查檔案大小（過大可能超時）
- 檢查 API 額度

### Embedding 失敗
- 程式會自動重試 3 次
- 檢查網路連線
- 檢查 API rate limits

## 測試範例

已包含測試檔案：`qa_data/test_company_faq.md`

執行測試：
```bash
uv run python build_vectordb.py
```

預期輸出：
```
找到 1 個 Markdown 檔案
  - test_company_faq.md

處理 test_company_faq.md...
  test_company_faq.md: [████████████] 2543/2543 chars (100%)
  ✓ 完成分塊：9 個 chunks
  ✓ 儲存 chunks：chunks/test_company_faq.json
  生成 embeddings：9 個 chunks...
    Batch 1/1 完成
  ✓ Embeddings 完成

建立 FAISS 向量資料庫...
總共 9 個 chunks
Embedding dimension: 3072
✓ Index 儲存至：faiss_db/index.faiss
✓ Metadata 儲存至：faiss_db/metadata.json

打包中...
✓ 向量資料庫已打包：faiss_db.zip

下一步：手動上傳 faiss_db.zip 到 Google Drive 並設為公開
```
