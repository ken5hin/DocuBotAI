from typing import List, Dict
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from config import EMBEDDING_MODEL, CHAT_MODEL, VECTOR_STORE_SIMILARITY_SEARCH_K

class ChatManager:
    def __init__(self):
        self.vector_store = None
        self.chat_chain = None
        
    def initialize_vector_store(self, documents: List):
        """Initialize FAISS vector store with documents"""
        self.vector_store = FAISS.from_documents(documents, EMBEDDING_MODEL)
        
        # Create a custom prompt template
        prompt_template = """You are a helpful AI assistant that answers questions based on the provided document context.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer based on the context, explain what information you would need to provide a better answer.
        
        Context: {context}
        
        Chat History: {chat_history}
        Human: {question}
        Assistant: Let me help you with that question. """

        CUSTOM_PROMPT = PromptTemplate.from_template(prompt_template)
        
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=CHAT_MODEL,
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": VECTOR_STORE_SIMILARITY_SEARCH_K,
                    "score_threshold": 0.5
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
