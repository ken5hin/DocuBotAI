from typing import List, Dict
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from config import VECTOR_STORE_SIMILARITY_SEARCH_K, get_llm_config

class ChatManager:
    def __init__(self):
        self.vector_store = None
        self.chat_chain = None
        self.current_provider = None
        
    def initialize_vector_store(self, documents: List, provider="gemini"):
        """Initialize FAISS vector store with documents"""
        if self.current_provider != provider:
            self.current_provider = provider
            models = get_llm_config(provider)
            self.vector_store = FAISS.from_documents(documents, models["embedding_model"])
        
        # Create a custom prompt template
        prompt_template = """You are a helpful AI assistant analyzing documents. Your task is to provide accurate answers based on the given document content.

        Here is the relevant context from the document:
        {context}

        Previous conversation:
        {chat_history}

        Current question: {question}

        Instructions:
        1. Use ONLY the information from the provided context to answer the question
        2. If the context doesn't contain enough information, explain specifically what's missing
        3. Be concise but thorough in your response
        4. If you quote from the document, use the exact words

        Assistant: Based on the document content, """

        CUSTOM_PROMPT = PromptTemplate.from_template(prompt_template)
        
        models = get_llm_config(self.current_provider)
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=models["chat_model"],
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": VECTOR_STORE_SIMILARITY_SEARCH_K,
                    "score_threshold": 0.3
                }
            ),
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
            return_source_documents=True,
            verbose=True
        )

    def get_response(self, question: str, chat_history: List) -> str:
        """Get AI response for user question"""
        if not self.chat_chain:
            raise ValueError("Chat chain not initialized. Please process a document first.")
        
        try:
            result = self.chat_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            
            if not result["answer"].strip():
                return "I apologize, but I couldn't find a relevant answer in the document. Could you please rephrase your question or ask about a different topic from the document?"
            
            return result["answer"]
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I encountered an error while processing your question. Please try asking in a different way or upload a different document if the issue persists."

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
