"""
GPT-5 API 程式設計助手 - Part B: RAG 對話機器人

本地執行:
    cd Week7
    uv run python rag_chatbot.py

Google Colab 執行:
    # 1. 安裝套件
    !pip install openai faiss-cpu numpy gradio

    # 2. 下載向量資料庫
    GDRIVE_LINK = "YOUR_GDRIVE_LINK_HERE"
    !gdown --fuzzy {GDRIVE_LINK}
    !unzip -o faiss_db.zip

    # 3. 設定 API key
    from google.colab import userdata
    import os
    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

    # 4. 執行程式
    !python rag_chatbot.py
"""

import os
import json
import numpy as np
import faiss
from openai import OpenAI
import gradio as gr
from typing import List, Dict, Tuple, Optional

# ===== 全域設定 =====
GDRIVE_FILE_ID = "YOUR_FILE_ID_HERE"  # 待填入
FAISS_DB_PATH = "faiss_db"
MAX_TOOL_ITERATIONS = 5

# System prompt for the chatbot
SYSTEM_PROMPT = """你是一位專業的 GPT-5 Response API 程式設計助手。

你的專長領域:
- GPT-5 Response API 的使用方式
- Function calling 和 custom tools
- Reasoning effort 控制 (minimal, low, medium, high)
- Text verbosity 設定
- Tool calling 的最佳實踐

回答風格:
- 友善、專業、有耐心
- 提供清楚的程式碼範例
- 解釋技術概念時用簡單易懂的方式
- 如果不確定答案,誠實告知並建議查閱官方文件

重要: 當你需要查詢 GPT-5 API 相關資料時,請使用 search_chunks 工具搜尋文件內容。"""


# ===== 1. 環境偵測與初始化 =====

def is_colab() -> bool:
    """偵測是否在 Google Colab 環境"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_environment() -> str:
    """
    設定環境並取得 API key

    Returns:
        str: OpenAI API key
    """
    if is_colab():
        from google.colab import userdata
        api_key = userdata.get('OPENAI_API_KEY')
        print("✓ Colab 環境: 從 userdata 載入 API key")
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        print("✓ 本地環境: 從環境變數載入 API key")

    if not api_key:
        raise ValueError("未找到 OPENAI_API_KEY,請確認環境設定")

    return api_key


def download_and_extract_faiss(gdrive_url: Optional[str] = None):
    """
    在 Colab 環境下載並解壓縮 FAISS 資料庫
    本地環境跳過此步驟

    Args:
        gdrive_url: Google Drive 分享連結 (可選)
    """
    if not is_colab():
        print("✓ 本地環境: 跳過下載,直接使用本地 faiss_db")
        return

    if not gdrive_url or gdrive_url == "YOUR_GDRIVE_LINK_HERE":
        print("⚠️  警告: 尚未設定 Google Drive 連結")
        print("請在程式碼中設定 GDRIVE_FILE_ID 或傳入 gdrive_url 參數")
        return

    print("⬇️  Colab 環境: 下載 faiss_db.zip...")
    os.system(f'gdown --fuzzy "{gdrive_url}"')

    print("📦 解壓縮 faiss_db.zip...")
    os.system('unzip -o faiss_db.zip')

    print("✓ FAISS 資料庫準備完成")


# ===== 2. FAISS 載入 =====

def load_vectorstore() -> Tuple[faiss.Index, List[Dict], OpenAI]:
    """
    載入 FAISS 向量資料庫

    Returns:
        Tuple[faiss.Index, List[Dict], OpenAI]: (FAISS index, metadata, OpenAI client)
    """
    print(f"📂 載入向量資料庫: {FAISS_DB_PATH}/")

    # 載入 FAISS index
    index_path = os.path.join(FAISS_DB_PATH, "index.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到 FAISS index: {index_path}")

    index = faiss.read_index(index_path)
    print(f"  ✓ FAISS index 載入完成: {index.ntotal} 個向量")

    # 載入 metadata
    metadata_path = os.path.join(FAISS_DB_PATH, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"找不到 metadata: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"  ✓ Metadata 載入完成: {len(metadata)} 個 chunks")

    # 初始化 OpenAI client
    client = OpenAI()

    return index, metadata, client


# ===== 3. RAG 檢索函數 =====

def search_chunks(query: str, index: faiss.Index, metadata: List[Dict],
                 client: OpenAI, k: int = 5) -> List[Dict]:
    """
    使用 FAISS 搜尋與問題最相關的 chunks

    Args:
        query: 使用者問題
        index: FAISS index
        metadata: Chunk metadata
        client: OpenAI client
        k: 回傳結果數量

    Returns:
        List[Dict]: 相關的 chunks
    """
    # 將 query 轉成 embedding
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[query]
    )
    query_embedding = response.data[0].embedding

    # FAISS 搜尋
    query_vector = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_vector, k)

    # 取得相關 chunks
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(metadata):
            chunk = metadata[idx].copy()
            chunk['distance'] = float(distance)
            results.append(chunk)

    return results


def format_chunks_for_llm(chunks: List[Dict]) -> str:
    """
    格式化 chunks 成適合 LLM 閱讀的格式

    Args:
        chunks: 搜尋結果 chunks

    Returns:
        str: 格式化後的文字
    """
    formatted_parts = []

    for i, chunk in enumerate(chunks, 1):
        source = chunk.get('source_file', 'Unknown')
        text = chunk.get('text', '')

        formatted_parts.append(f"[文件 {i}: {source}]\n{text}")

    return "\n\n---\n\n".join(formatted_parts)


# ===== 4. GPT-5 對話引擎 =====

def convert_gradio_history_to_openai(history: List[Tuple[str, str]]) -> List[Dict]:
    """
    轉換 Gradio history 格式為 OpenAI messages 格式

    Args:
        history: Gradio history [(user_msg, bot_msg), ...]

    Returns:
        List[Dict]: OpenAI messages 格式
    """
    messages = []

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    return messages


def chat_with_rag(user_message: str, gradio_history: List[Tuple[str, str]],
                 index: faiss.Index, metadata: List[Dict],
                 client: OpenAI) -> str:
    """
    主要對話函數,整合 RAG 檢索和 GPT-5 回答

    Args:
        user_message: 使用者輸入
        gradio_history: Gradio 對話歷史
        index: FAISS index
        metadata: Chunk metadata
        client: OpenAI client

    Returns:
        str: AI 回答
    """
    # 1. 轉換對話歷史
    input_messages = convert_gradio_history_to_openai(gradio_history)
    input_messages.append({"role": "user", "content": user_message})

    # 2. 定義 search_chunks tool
    tools = [{
        "type": "function",
        "name": "search_chunks",
        "description": "搜尋 GPT-5 Response API 官方文件內容。當使用者詢問關於 GPT-5 API、function calling、tool calling、reasoning effort、verbosity 等技術問題時,請使用此工具搜尋相關文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜尋關鍵字或問題,例如: 'custom tools', 'reasoning effort', 'function calling example'"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        },
        "strict": True
    }]

    # 3. Tool calling loop
    iteration = 0

    while iteration < MAX_TOOL_ITERATIONS:
        print(f"\n🔄 Tool calling iteration {iteration + 1}")

        # 呼叫 GPT-5
        response = client.responses.create(
            model="gpt-5",
            reasoning={"effort": "low"},
            instructions=SYSTEM_PROMPT,
            input=input_messages,
            tools=tools
        )

        # 將 response.output 的每個 item 加到 input_messages
        # 重要: 不要直接 += response.output，要逐個 append
        for item in response.output:
            input_messages.append(item)

        # 檢查是否有 function calls 並執行
        has_tool_calls = False

        for item in response.output:
            if item.type == "function_call":
                has_tool_calls = True
                function_name = item.name
                call_id = item.call_id

                print(f"  📞 GPT-5 呼叫工具: {function_name}")
                print(f"     參數: {item.arguments}")

                if function_name == "search_chunks":
                    # 解析參數
                    args = json.loads(item.arguments)
                    query = args.get("query", "")

                    # 執行檢索 (固定使用 k=5)
                    print(f"  🔍 執行檢索: query='{query}'")
                    chunks = search_chunks(query, index, metadata, client, k=5)

                    # 格式化結果
                    formatted_result = format_chunks_for_llm(chunks)

                    print(f"  ✓ 找到 {len(chunks)} 個相關文件片段")

                    # 加入 function_call_output
                    input_messages.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": formatted_result
                    })

        # 如果沒有 tool calls,結束循環
        if not has_tool_calls:
            print("  ✓ GPT-5 完成回答,無需更多工具")
            break

        iteration += 1

    if iteration >= MAX_TOOL_ITERATIONS:
        print(f"⚠️  警告: 達到最大迭代次數 ({MAX_TOOL_ITERATIONS})")

    # 4. 提取最終回答
    final_answer = response.output_text

    return final_answer


# ===== 5. Gradio 介面 =====

def create_chatbot_interface(index: faiss.Index, metadata: List[Dict],
                            client: OpenAI) -> gr.Interface:
    """
    建立 Gradio chatbot 介面

    Args:
        index: FAISS index
        metadata: Chunk metadata
        client: OpenAI client

    Returns:
        gr.Interface: Gradio 介面
    """
    def chat_wrapper(message: str, history: List[Tuple[str, str]]) -> str:
        """Gradio ChatInterface 的 wrapper"""
        return chat_with_rag(message, history, index, metadata, client)

    # 建立 ChatInterface
    demo = gr.ChatInterface(
        fn=chat_wrapper,
        title="🤖 GPT-5 API 程式設計助手",
        description="""
我可以協助你學習和使用 **GPT-5 Response API**!

**我的專長領域:**
- Function calling 和 custom tools 的使用
- Reasoning effort 控制 (minimal, low, medium, high)
- Text verbosity 設定
- Tool calling 最佳實踐
- 程式碼範例和實作建議

**提示:** 我會自動搜尋官方文件來回答你的問題,你可以問我任何關於 GPT-5 API 的技術細節!
        """.strip(),
        examples=[
            "如何使用 custom tools?",
            "function calling 的完整流程是什麼?",
            "reasoning effort 的 minimal 和 low 有什麼差別?",
            "請給我一個 tool calling loop 的程式碼範例",
            "如何設定 verbosity 來控制輸出長度?"
        ],
        theme=gr.themes.Soft()
    )

    return demo


# ===== 6. 主程式執行 =====

def main():
    """主程式進入點"""
    print("=" * 60)
    print("🚀 GPT-5 API 程式設計助手 - Part B: RAG 對話機器人")
    print("=" * 60)

    try:
        # 1. 環境偵測與初始化
        print("\n[1/4] 環境設定")
        api_key = setup_environment()

        # 2. 下載 FAISS (僅 Colab)
        print("\n[2/4] FAISS 資料庫準備")
        download_and_extract_faiss()

        # 3. 載入向量資料庫
        print("\n[3/4] 載入向量資料庫")
        index, metadata, client = load_vectorstore()

        # 4. 啟動 Gradio
        print("\n[4/4] 啟動 Gradio 介面")
        demo = create_chatbot_interface(index, metadata, client)

        print("\n✨ 系統準備完成!")
        print("=" * 60)

        # 啟動介面
        demo.launch(
            share=True,  # 產生 share link (Colab 需要)
            server_port=7860,
            show_error=True
        )

    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        raise


if __name__ == "__main__":
    main()
