from typing import List, Dict
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from config import EMBEDDING_MODEL, CHAT_MODEL, VECTOR_STORE_SIMILARITY_SEARCH_K

class ChatManager:
    def __init__(self):
        self.vector_store = None
        self.chat_chain = None

    def initialize_vector_store(self, documents: List):
        """Initialize FAISS vector store with documents"""
        self.vector_store = FAISS.from_documents(documents, EMBEDDING_MODEL)
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=CHAT_MODEL,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": VECTOR_STORE_SIMILARITY_SEARCH_K}
            ),
        )

    def get_response(self, question: str, chat_history: List) -> str:
        """Get AI response for user question"""
        if not self.chat_chain:
            raise ValueError("Chat chain not initialized. Please process a document first.")
        
        result = self.chat_chain({"question": question, "chat_history": chat_history})
        return result["answer"]

    @staticmethod
    def format_chat_history(messages: List[Dict]) -> List[tuple]:
        """Format chat history for LangChain"""
        return [(m["user"], m["assistant"]) for m in messages]
