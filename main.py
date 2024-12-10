import streamlit as st
from utils.document_processor import DocumentProcessor
from utils.chat_manager import ChatManager
import time

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()

def clear_chat_history():
    """Clear chat history"""
    st.session_state.messages = []

def main():
    st.set_page_config(
        page_title="ACCA Knowledge Assistant",
        page_icon="💬",
        layout="wide"
    )

    initialize_session_state()

    st.title("📚 ACCA Knowledge Assistant")

    # Sidebar for model selection and document upload
    with st.sidebar:
        # Add ACCA logo at the top
        st.image("image.png", width=150, use_column_width=False)
        
        st.header("Settings")
        
        # Model selection
        model_provider = st.selectbox(
            "Select LLM Provider",
            options=["gemini", "openai"],
            key="model_provider"
        )
        
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload your document (PDF or TXT)",
            type=["pdf", "txt"]
        )

        if uploaded_file:
            with st.spinner("Processing document..."):
                # Validate file
                is_valid, error_message = st.session_state.document_processor.validate_file(
                    uploaded_file
                )
                
                if not is_valid:
                    st.error(error_message)
                    return

                # Process document
                try:
                    text = st.session_state.document_processor.extract_text(uploaded_file)
                    documents = st.session_state.document_processor.process_text(text)
                    st.session_state.chat_manager.initialize_vector_store(
                        documents,
                        provider=st.session_state.model_provider
                    )
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    return

        st.button("Clear Chat History", on_click=clear_chat_history)

    # Main chat interface
    chat_container = st.container()
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your document"):
            if not uploaded_file:
                st.error("Please upload a document first!")
                return

            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        chat_history = ChatManager.format_chat_history(
                            st.session_state.messages[:-1]
                        )
                        response = st.session_state.chat_manager.get_response(
                            prompt, chat_history
                        )
                        message_placeholder.write(response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        message_placeholder.error(error_message)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_message}
                        )

if __name__ == "__main__":
    main()
