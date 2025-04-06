import os
import json
import asyncio
from dotenv import load_dotenv
from supabase import create_client
from openai import AsyncOpenAI
from pymongo import MongoClient
from typing import List

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB_NAME]
structured_collection = mongo_db["structured_data"]

def log(message: str, level="LOG"):
    """Log messages with different levels."""
    print(f"[{level}] {message}")

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    log("Generating embedding for query...")
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        log(f"Error getting embedding: {e}", "ERROR")
        return [0] * 1536  # Return zero vector on error

async def retrieve_chunks(url: str):
    """Retrieve webpage chunks from Supabase."""
    log(f"Fetching chunks for URL: {url}")
    result = supabase.from_("site_pages").select("content").eq("url", url).execute()
    
    if not result.data:
        log("No chunks found.", "WARNING")
        return []
    
    chunks = [doc["content"] for doc in result.data]
    log(f"Retrieved {len(chunks)} chunks.")
    return chunks

async def extract_structured_data(text: str):
    """Use OpenAI to extract structured data."""
    log("Extracting structured data from content...")
    
    if len(text) > 100_000:
        log(f"Warning: Text is too long ({len(text)} characters), truncating...", "WARNING")
        text = text[:100_000]

    prompt = f"""
    Extract structured data from the following text and return as JSON:
    {text}
    """

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}  # FIXED format
        )
        
        raw_response = response.choices[0].message.content
        log(f"Raw OpenAI response: {raw_response[:500]}...", "DEBUG")  # Show first 500 chars
        
        structured_json = json.loads(raw_response)
        log("Structured data extracted successfully.")
        print(json.dumps(structured_json, indent=4))  # Print structured data to console
        return structured_json

    except json.JSONDecodeError:
        log("Error: OpenAI response is not valid JSON!", "ERROR")
        return {}

    except Exception as e:
        log(f"Error extracting structured data: {e}", "ERROR")
        return {}

async def process_webpage(url: str):
    """Main function to process a webpage."""
    log(f"Processing webpage: {url}")
    
    chunks = await retrieve_chunks(url)
    if not chunks:
        log("No content to process.", "WARNING")
        return
    
    combined_text = "\n".join(chunks)
    log(f"Combined text length: {len(combined_text)} characters")

    structured_data = await extract_structured_data(combined_text)
    if structured_data:
        log("Saving structured data to MongoDB...")
        structured_collection.insert_one({"url": url, "data": structured_data})
        log("Structured data saved successfully.")
    else:
        log("No structured data extracted.", "WARNING")

if __name__ == "__main__":
    test_url = "https://www.ox.ac.uk/admissions/graduate/student-life/sport-arts-and-societies/initial"
    asyncio.run(process_webpage(test_url))
