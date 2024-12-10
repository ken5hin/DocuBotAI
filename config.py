import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Configuration settings
ALLOWED_EXTENSIONS = ['.pdf', '.txt']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def get_llm_config(provider="gemini"):
    """Get LLM configuration based on provider"""
    if provider == "openai":
        return {
            "embedding_model": OpenAIEmbeddings(),
            "chat_model": ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=2048,
            )
        }
    else:  # gemini
        return {
            "embedding_model": GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            "chat_model": ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.3,
                convert_system_message_to_human=True,
                top_p=0.8,
                max_output_tokens=2048,
            )
        }

# Default to Gemini models
default_models = get_llm_config("gemini")
EMBEDDING_MODEL = default_models["embedding_model"]
CHAT_MODEL = default_models["chat_model"]

# Vector store settings
VECTOR_STORE_SIMILARITY_SEARCH_K = 3

# Chat settings
MAX_HISTORY_LENGTH = 5
