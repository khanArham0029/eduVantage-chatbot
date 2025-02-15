from chat_processor import get_chat_response
from chat_history import save_chat_session, load_chat_session
from chat_mongo import get_chat_collection
from chat_cv_processor import process_cv
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat messages and returns a chatbot response."""
    data = request.json
    user_id = data.get("user_id")
    user_input = data.get("message")

    if not user_id or not user_input:
        return jsonify({"error": "Missing user_id or message"}), 400

    response = get_chat_response(user_id, user_input)

    return jsonify({"response": response})

@app.route("/upload_cv", methods=["POST"])
def upload_cv():
    """Processes an uploaded CV and extracts text."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    text = process_cv(file)

    return jsonify({"extracted_text": text})

if __name__ == "__main__":
    app.run(debug=True)
