from typing import List, Dict
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
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
            return_source_documents=True,
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
        chat_history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                chat_history.append((
                    messages[i]["content"],
                    messages[i + 1]["content"]
                ))
        return chat_history
