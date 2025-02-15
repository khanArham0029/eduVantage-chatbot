from langchain.schema import AIMessage, HumanMessage
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB connection string
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client.eduVantage
chat_sessions = db.chat_sessions

def save_chat_session(user_id, user_input, response_content):
    """Save or update a user's chat session in MongoDB."""
    chat_sessions.update_one(
        {"user_id": user_id},
        {"$push": {"history": {"$each": [{"role": "user", "content": user_input}, {"role": "ai", "content": response_content}]}}},
        upsert=True
    )

def load_chat_session(user_id):
    """Retrieve a user's chat session from MongoDB."""
    session = chat_sessions.find_one({"user_id": user_id})
    if session and "history" in session:
        return [AIMessage(content=msg["content"]) if msg["role"] == "ai" else HumanMessage(content=msg["content"]) for msg in session["history"]]
    return []
