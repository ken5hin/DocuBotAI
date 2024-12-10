import PyPDF2
from typing import List, Tuple
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import io
import os
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
            is_separator_regex=False
        )

    def validate_file(self, file) -> Tuple[bool, str]:
        """Validate uploaded file"""
        if file is None:
            return False, "No file uploaded"
        
        # Check file size
        file_size = len(file.getvalue())
        if file_size > MAX_FILE_SIZE:
            return False, f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit"
        
        # Check file extension
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type. Please upload: {', '.join(ALLOWED_EXTENSIONS)}"
        
        return True, ""

    def extract_text(self, uploaded_file) -> str:
        """Extract text from uploaded document"""
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.pdf':
            return self._extract_from_pdf(uploaded_file)
        elif file_extension == '.txt':
            return self._extract_from_txt(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _extract_from_pdf(self, file) -> str:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _extract_from_txt(self, file) -> str:
        return file.getvalue().decode('utf-8')

    def process_text(self, text: str) -> List[Document]:
        """Split text into chunks and create documents"""
        if not text.strip():
            raise ValueError("Empty document content")
        
        documents = self.text_splitter.create_documents([text])
        if not documents:
            raise ValueError("No documents created after processing")
            
        # Add logging to verify document content
        st.info(f"Processed {len(documents)} document chunks")
        
        # Debug logging for document content
        for i, doc in enumerate(documents):
            st.debug(f"Chunk {i+1} (length: {len(doc.page_content)}): {doc.page_content[:100]}...")
            
        return documents
