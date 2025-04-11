from __future__ import annotations as _annotations

import os
import asyncio
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

import logfire
from openai import AsyncOpenAI
from supabase import create_client, Client
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import re

# --- Load Environment Variables ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --- Initialize Clients ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = AsyncOpenAI()

# --- Pydantic AI Model Setup ---
llm = os.getenv("LLM_MODEL", "gpt-4o")
model = OpenAIModel(llm)
logfire.configure(send_to_logfire="if-token-present")

# --- Dependency Injection ---
@dataclass
class UniversityAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

# --- System Prompt for Agent ---
system_prompt = """
You are a university assistant. Focus only on the university mentioned in the user's query.
Your job is to find accurate information using tools you have access to.
Never make up an answer — use the content retrieved from Supabase.

The user might ask general or specific questions about a university's admission, courses, facilities, etc.
"""

# --- Initialize Agent ---
university_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=UniversityAIDeps,
    retries=2,
)

# --- Utility: Get Embedding ---
async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    print("[LOG] Generating embedding...")
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[ERROR] Failed to get embedding: {e}")
        return [0.0] * 1536

# --- Tool: Retrieve Relevant Info from Supabase ---
@university_agent.tool
async def retrieve_university_info(ctx: RunContext[UniversityAIDeps], query: str) -> str:
    print("[LOG] Retrieving relevant university chunks from Supabase...")

    embedding = await get_embedding(query, ctx.deps.openai_client)

    # Attempt to extract university name from query (simple keyword match)
    # Example assumes university name is mentioned explicitly like "about NUST"
    university_keywords = ["nust", "fast", "lums", "comsats", "air", "iba", "giki","stanford","oxford", "mit", "caltech", "harvard", "cambridge", "berkeley", "princeton"]
    matched_uni = next(
        (uni for uni in university_keywords if re.search(rf"\b{re.escape(uni)}\b", query.lower())),
    None
    )
    if not matched_uni:
        return "Please mention a specific university in your query (e.g., NUST, FAST, LUMS)."

    try:
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': embedding,
                'match_count': 5,
                'filter': {'university_name': matched_uni.lower()}
            }
        ).execute()

        print(f"[LOG] Supabase RPC call completed. Matching chunks: {len(result.data) if result.data else 0}")

        if not result.data:
            print(f"[LOG] No Supabase results found. Falling back to Tavily.")
            return await tavily_web_search(ctx, query)

        # Filter manually by university name in URL (post-hoc since filter may not work in RPC)
        filtered = [
            doc for doc in result.data
            if "metadata" in doc and doc["metadata"].get("university_name", "").lower() == matched_uni.lower()
        ]

        if not filtered:
            print(f"[LOG] No relevant filtered chunks. Falling back to Tavily.")
            return await tavily_web_search(ctx, query)

        print(f"[LOG] {len(filtered)} chunks matched {matched_uni.upper()}.")

        content = []
        for doc in filtered:
            content.append(f"# {doc['title']}\n\n{doc['summary'] or doc['content'][:300]}...")

        return "\n\n---\n\n".join(content)

    except Exception as e:
        print(f"[ERROR] Failed to retrieve university info: {e}")
        return "There was an error fetching data. Please try again."

# --- Tool: List All Available Universities ---
@university_agent.tool
async def list_universities(ctx: RunContext[UniversityAIDeps]) -> List[str]:
    print("[LOG] Fetching available universities (from university_name column)...")
    try:
        result = ctx.deps.supabase.from_("site_pages") \
            .select("university_name") \
            .execute()

        if not result.data:
            return []

        universities = set()
        for row in result.data:
            uni_name = row.get("university_name")
            if uni_name:
                universities.add(uni_name.strip().upper())

        print(f"[LOG] Found universities: {sorted(universities)}")
        return sorted(universities)
    except Exception as e:
        print(f"[ERROR] Failed to list universities: {e}")
        return []


@university_agent.tool
async def tavily_web_search(ctx: RunContext[UniversityAIDeps], query: str) -> str:
    import requests

    print("[LOG] Calling Tavily for web search...")
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "search_depth": "basic",
        "chunks_per_source": 3,
        "max_results": 1,
        "days": 7,
        "include_answer": True,
        "include_raw_content": False,
        "include_images": False,
        "include_image_descriptions": False,
        "include_domains": [],
        "exclude_domains": []
    }

    try:
        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        answer = result.get("answer", "No answer found.")
        url = result["results"][0]["url"] if result["results"] else "N/A"
        title = result["results"][0]["title"] if result["results"] else "N/A"
        content = result["results"][0]["content"] if result["results"] else answer

        print(f"[LOG] Tavily search completed. Answer: {answer[:60]}...")

        # Generate and store embedding
        embedding = await get_embedding(answer, ctx.deps.openai_client)

        insert_data = {
            "url": url,
            "title": title,
            "content": content,
            "summary": answer,
            "embedding": embedding,
            "chunk_number": 0
        }

        print("[LOG] Caching result in Supabase...")
        exists = ctx.deps.supabase.from_("site_pages").select("id").eq("url", url).execute()
        if not exists.data:
            ctx.deps.supabase.from_("site_pages").insert(insert_data).execute()

        return answer

    except Exception as e:
        print(f"[ERROR] Tavily web search failed: {e}")
        return "There was an error retrieving the latest web result. Please try again later."


# --- Main Runner ---
async def main():
    print("[LOG] Starting University Agent...")
    ctx = UniversityAIDeps(supabase=supabase, openai_client=openai_client)

    query = "Tell me about the program Aeronautics and Astronautics at mit. Give me details about the curriculum, admission requirements, and duration, deadlines"
    print(f"[LOG] Running query: {query}")

    response = await university_agent.run(query, deps=ctx)
    print("\n[RESULT]")
    print(response)

# --- Run if main ---
if __name__ == "__main__":
    asyncio.run(main())