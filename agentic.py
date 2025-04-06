import os
import json
import sys
import asyncio
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()  # Load environment variables from .env file

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk using OpenAI."""
    # Get title and summary using OpenAI
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding using OpenAI
    embedding = await get_embedding(chunk)
    
    # Create dynamic metadata
    parsed_url = urlparse(url)
    metadata = {
        "source": parsed_url.netloc,  # Generic source from URL domain
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": parsed_url.path,
        "course_identifier": parsed_url.path.split('/')[1] if '/' in parsed_url.path else "unknown"
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title/summary using OpenAI."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
        return {"title": "Error", "summary": "Processing error"}

async def get_embedding(text: str) -> List[float]:
    """Generate embeddings with OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error: {e}")
        return [0] * 1536  # Return zero vector on error

# Modified chunking function without code blocks
def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        if '\n\n' in text[start:end]:
            last_break = text.rfind('\n\n', start, end)
            if last_break != -1 and (last_break - start) > 0.3 * chunk_size:
                end = last_break + 2  
        else:
            last_period = text.rfind('. ', start, end)
            if last_period != -1 and (last_period - start) > 0.3 * chunk_size:
                end = last_period + 1  

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None
    
async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    chunks = chunk_text(markdown)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)

async def main():
    if len(sys.argv) != 3:
        print("Usage: python agentic.py <url> <markdown_file_path>")
        return

    url = sys.argv[1]
    file_path = sys.argv[2]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    await process_and_store_document(url, markdown_content)
    print(f"✅ Finished processing: {file_path}")


if __name__ == "__main__":
    asyncio.run(main())
