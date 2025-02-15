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

def get_chat_collection():
    """Return the chat sessions collection."""
    return chat_sessions
