import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Configuration settings
ALLOWED_EXTENSIONS = ['.pdf', '.txt']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Model configurations
EMBEDDING_MODEL = OpenAIEmbeddings()
CHAT_MODEL = ChatOpenAI(temperature=0.7)

# Vector store settings
VECTOR_STORE_SIMILARITY_SEARCH_K = 3

# Chat settings
MAX_HISTORY_LENGTH = 5
