import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration settings
ALLOWED_EXTENSIONS = ['.pdf', '.txt']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Model configurations
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
CHAT_MODEL = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    convert_system_message_to_human=True
)

# Vector store settings
VECTOR_STORE_SIMILARITY_SEARCH_K = 3

# Chat settings
MAX_HISTORY_LENGTH = 5
