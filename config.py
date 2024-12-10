import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

# Configuration settings
ALLOWED_EXTENSIONS = ['.pdf', '.txt']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Get and validate Google API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please make sure GOOGLE_API_KEY is set in your environment variables.")
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

try:
    # Configure Google API
    genai.configure(api_key=GOOGLE_API_KEY)

    # Model configurations
    EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    CHAT_MODEL = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        convert_system_message_to_human=True,
        top_p=0.8,
        max_output_tokens=2048,
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    st.error(f"Error initializing Google AI models: {str(e)}")
    raise

# Vector store settings
VECTOR_STORE_SIMILARITY_SEARCH_K = 3

# Chat settings
MAX_HISTORY_LENGTH = 5
