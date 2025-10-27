#!/usr/bin/env python3
"""
Part A: 建立向量資料庫
使用 GPT-5 智能分塊 + OpenAI Embeddings + FAISS
"""

import os
import json
import time
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import numpy as np
import faiss
from openai import OpenAI

# 初始化 OpenAI client（自動從環境變數讀取 OPENAI_API_KEY）
client = OpenAI()

# 設定目錄
QA_DATA_DIR = "qa_data"
CHUNKS_DIR = "chunks"
FAISS_DIR = "faiss_db"
OUTPUT_ZIP = "faiss_db.zip"

# 設定
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_BATCH_SIZE = 100
MAX_WORKERS = 5  # 並行處理的檔案數
RETRY_ATTEMPTS = 3


def ensure_directories():
    """確保必要的目錄存在"""
    Path(CHUNKS_DIR).mkdir(exist_ok=True)
    Path(FAISS_DIR).mkdir(exist_ok=True)


def read_markdown_file(file_path: str) -> str:
    """讀取 Markdown 檔案"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_with_gpt5(md_file: str, content: str) -> List[Dict]:
    """
    使用 GPT-5 進行智能語義分塊
    使用 custom tool calling 遞迴循環
    """
    print(f"\n處理 {md_file}...")

    total_chars = len(content)
    processed_chars = 0
    chunks = []

    # 定義 custom tool
    tools = [{
        "type": "custom",
        "name": "save_chunk",
        "description": "儲存一個語義完整的 Q&A chunk。每個 chunk 應該包含完整的問答對或邏輯段落。處理完一個 chunk 後會繼續處理下一個。"
    }]

    # 初始 input
    input_messages = [
        {
            "role": "user",
            "content": f"將此文件分割成語義 chunks。使用 save_chunk tool 儲存每個 chunk。\n\n文件內容：\n{content}"
        }
    ]

    try:
        # 遞迴循環：持續呼叫 GPT-5 直到沒有更多 tool calls
        max_iterations = 50  # 防止無限循環
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # 呼叫 GPT-5
            response = client.responses.create(
                model="gpt-5",
                reasoning={"effort": "minimal"},
                instructions="""你是文件分塊助手。你的任務是將文件切分成語義完整的 chunks,而不是總結或改寫內容。

重要規則:
1. 保持原文內容完全不變 - 不要改寫、不要總結、不要重組
2. 直接複製原文的文字,包括標題、格式、換行
3. 按照語義邏輯切分:完整的段落、完整的 Q&A 對、完整的程式碼範例
4. 每個 chunk 應該是一個獨立、完整的概念單元
5. 使用 save_chunk tool 儲存每個 chunk 的原文內容
6. 處理完一個 chunk 後繼續處理下一個,直到整份文件處理完畢

記住:你是在「切分」而不是「改寫」,保持原文完整性是最重要的。""",
                input=input_messages,
                tools=tools
            )

            # 收集這一輪的 tool calls
            has_tool_calls = False

            for item in response.output:
                if item.type == "custom_tool_call":
                    has_tool_calls = True
                    chunk_text = item.input
                    processed_chars += len(chunk_text)

                    # 更新進度條
                    progress_pct = min(100, int(100 * processed_chars / total_chars))
                    bar_length = 40
                    filled = int(bar_length * progress_pct / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)

                    print(f"\r  {md_file}: [{bar}] {processed_chars}/{total_chars} chars ({progress_pct}%)", end="", flush=True)

                    chunks.append({
                        "chunk_id": item.call_id,
                        "text": chunk_text,
                        "source_file": md_file,
                        "chunk_index": len(chunks),
                        "char_count": len(chunk_text)
                    })

            if not has_tool_calls:
                # 沒有更多 tool calls，結束
                break

            # 把這一輪的 response output items 加到 input
            for item in response.output:
                input_messages.append(item)

            # 為每個 tool call 準備 output
            for item in response.output:
                if item.type == "custom_tool_call":
                    input_messages.append({
                        "type": "custom_tool_call_output",
                        "call_id": item.call_id,
                        "output": "已儲存，繼續處理下一個 chunk"
                    })

        print()  # 換行
        print(f"  ✓ 完成分塊：{len(chunks)} 個 chunks（{iteration} 輪迭代）")
        return chunks

    except Exception as e:
        print(f"\n  ✗ GPT-5 分塊失敗：{e}")
        raise


def generate_embeddings(chunks: List[Dict]) -> List[List[float]]:
    """
    批量生成 embeddings
    自動重試機制
    """
    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []

    print(f"  生成 embeddings：{len(texts)} 個 chunks...")

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i+EMBEDDING_BATCH_SIZE]
        batch_num = i // EMBEDDING_BATCH_SIZE + 1
        total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        # 自動重試
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)

                print(f"    Batch {batch_num}/{total_batches} 完成", flush=True)
                break

            except Exception as e:
                if attempt == RETRY_ATTEMPTS - 1:
                    print(f"\n  ✗ Embedding 失敗（已重試 {RETRY_ATTEMPTS} 次）：{e}")
                    raise
                else:
                    wait_time = 2 ** attempt
                    print(f"    重試中... (等待 {wait_time}s)")
                    time.sleep(wait_time)

    print(f"  ✓ Embeddings 完成")
    return all_embeddings


def save_chunks_json(md_file: str, chunks: List[Dict]):
    """儲存 chunks 為 JSON"""
    output_path = Path(CHUNKS_DIR) / f"{Path(md_file).stem}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 儲存 chunks：{output_path}")


def process_single_file(md_file: str) -> Tuple[List[Dict], List[List[float]]]:
    """
    處理單一檔案：GPT-5 分塊 → 生成 embeddings
    """
    file_path = Path(QA_DATA_DIR) / md_file

    try:
        # 1. 讀取檔案
        content = read_markdown_file(file_path)

        # 2. GPT-5 智能分塊
        chunks = chunk_with_gpt5(md_file, content)

        # 3. 儲存 chunks JSON
        save_chunks_json(md_file, chunks)

        # 4. 生成 embeddings
        embeddings = generate_embeddings(chunks)

        return chunks, embeddings

    except Exception as e:
        print(f"\n✗ 處理 {md_file} 失敗：{e}")
        return [], []


def build_faiss_index(all_chunks: List[List[Dict]], all_embeddings: List[List[List[float]]]):
    """
    建立 FAISS 向量資料庫
    """
    print("\n" + "="*60)
    print("建立 FAISS 向量資料庫...")
    print("="*60)

    # 合併所有 chunks 和 embeddings
    merged_chunks = []
    merged_embeddings = []

    for chunks, embeddings in zip(all_chunks, all_embeddings):
        merged_chunks.extend(chunks)
        merged_embeddings.extend(embeddings)

    total_chunks = len(merged_chunks)
    print(f"總共 {total_chunks} 個 chunks")

    if total_chunks == 0:
        print("✗ 沒有 chunks 可以建立向量資料庫！")
        return

    # 轉換為 numpy array
    embeddings_array = np.array(merged_embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    print(f"Embedding dimension: {dimension}")

    # 建立 FAISS index（使用 L2 距離）
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # 儲存 index
    index_path = Path(FAISS_DIR) / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"✓ Index 儲存至：{index_path}")

    # 建立並儲存 metadata
    metadata = []
    for vector_idx, chunk in enumerate(merged_chunks):
        metadata.append({
            **chunk,
            "vector_index": vector_idx,
            "embedding_model": EMBEDDING_MODEL
        })

    metadata_path = Path(FAISS_DIR) / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✓ Metadata 儲存至：{metadata_path}")

    # 打包成 zip
    print("\n打包中...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(index_path, f"faiss_db/{index_path.name}")
        zipf.write(metadata_path, f"faiss_db/{metadata_path.name}")

    print(f"✓ 向量資料庫已打包：{OUTPUT_ZIP}")
    print(f"\n下一步：手動上傳 {OUTPUT_ZIP} 到 Google Drive 並設為公開")


def main():
    """主程式"""
    print("="*60)
    print("Part A: 建立向量資料庫")
    print("="*60)

    # 確保目錄存在
    ensure_directories()

    # 掃描 qa_data 目錄
    qa_data_path = Path(QA_DATA_DIR)
    if not qa_data_path.exists():
        print(f"\n✗ 錯誤：{QA_DATA_DIR}/ 目錄不存在！")
        print(f"請建立 {QA_DATA_DIR}/ 目錄並放入 Markdown 檔案")
        return

    md_files = list(qa_data_path.glob("*.md"))
    if not md_files:
        print(f"\n✗ 錯誤：{QA_DATA_DIR}/ 目錄中沒有 .md 檔案！")
        return

    print(f"\n找到 {len(md_files)} 個 Markdown 檔案")
    for f in md_files:
        print(f"  - {f.name}")

    # 並行處理所有檔案
    print("\n" + "="*60)
    print("並行處理中...")
    print("="*60)

    all_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任務
        futures = {
            executor.submit(process_single_file, f.name): f.name
            for f in md_files
        }

        # 收集結果
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                chunks, embeddings = future.result()
                if chunks and embeddings:
                    all_results.append((chunks, embeddings))
            except Exception as e:
                print(f"\n✗ {file_name} 處理失敗：{e}")

    # 建立 FAISS index
    if all_results:
        all_chunks = [r[0] for r in all_results]
        all_embeddings = [r[1] for r in all_results]
        build_faiss_index(all_chunks, all_embeddings)
    else:
        print("\n✗ 沒有成功處理的檔案！")

    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == "__main__":
    main()
