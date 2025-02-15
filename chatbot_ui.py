import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from chatbot_backend import process_cv, get_chat_response
import uuid  # Generate unique user IDs

# Streamlit UI Setup
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("🤖 AI Chatbot with CV Upload")

# Generate or retrieve user session ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())  # Assign a unique ID

user_id = st.session_state.user_id

# Sidebar for file upload
st.sidebar.header("Upload your CV (PDF)")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    st.sidebar.success("CV uploaded successfully!")
    extracted_text = process_cv(uploaded_file)
    st.sidebar.text_area("Extracted Text:", extracted_text, height=150)

# Initialize or load chat history
if "messages" not in st.session_state:
    from chatbot_backend import load_chat_session
    st.session_state.messages = load_chat_session(user_id)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.write(msg.content)

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Append user input to history
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Get chatbot response
    response = get_chat_response(user_id, user_input)

    # Append AI response to history
    st.session_state.messages.append(AIMessage(content=response))

    # Display AI response
    with st.chat_message("assistant"):
        st.write(response)
