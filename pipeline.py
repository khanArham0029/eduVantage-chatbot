import asyncio
import sys
import os

from agentic import process_and_store_document
from agentic_struct import process_webpage

async def run_pipeline(url: str, markdown_path: str):
    if not os.path.exists(markdown_path):
        print(f"❌ Error: Markdown file '{markdown_path}' does not exist.")
        return

    try:
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return

    print(f"\n📥 Loaded markdown from: {markdown_path}")
    print(f"🔗 Target URL: {url}\n")

    try:
        print("📄 Step 1: Summarizing and storing chunks in Supabase...")
        await process_and_store_document(url, markdown_text)
        print("✅ Step 1 complete!\n")
    except Exception as e:
        print(f"❌ Error during summarization & storage: {e}")
        return

    try:
        print("🏗️ Step 2: Structuring data using OpenAI and saving to MongoDB...")
        await process_webpage(url)
        print("✅ Step 2 complete!\n")
    except Exception as e:
        print(f"❌ Error during structuring: {e}")
        return

    print("🎉 All steps completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <url> <markdown_file_path>")
        sys.exit(1)

    url = sys.argv[1]
    markdown_path = sys.argv[2]

    asyncio.run(run_pipeline(url, markdown_path))
