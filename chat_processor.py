from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage
from chat_history import save_chat_session, load_chat_session

# Initialize chatbot
chatbot = ChatOllama(model="llama3.2")

def get_chat_response(user_id, user_input):
    """Generate a chatbot response and update history."""
    
    # Load previous chat history
    history = load_chat_session(user_id)

    # Convert history to LangChain message format
    chat_history = []
    for msg in history:
        chat_history.append(msg)  # Directly append AIMessage / HumanMessage

    # Append latest user input
    chat_history.append(HumanMessage(content=user_input))

    # Get chatbot response
    response = chatbot.invoke(chat_history)  

    # Save response to chat history
    save_chat_session(user_id, user_input, response.content)

    return response.content  # Ensure returning response as string