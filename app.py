import streamlit as st
from src.rag_pipeline import create_rag_pipeline, query_rag
import os

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("Chat with PDF")
st.subheader("Ask questions about the uploaded document")

# Sidebar for API key and info
with st.sidebar:
    st.write("**Model**: Gemini Pro")
    st.write("**Embedding**: Google embedding-001")
    st.write("**Vector DB**: Chroma")
    if os.path.exists("chunks/chunks.json"):
        import json
        with open("chunks/chunks.json", "r") as f:
            chunks = json.load(f)
        st.write(f"**Indexed Chunks**: {len(chunks)}")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

# Initialize RAG pipeline
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = create_rag_pipeline()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Query input and response
query = st.text_input("Your question:")
if query:
    with st.spinner("Generating response..."):
        result = query_rag(st.session_state.rag_chain, query)
        answer = result["answer"]
        sources = result["source_documents"]
        
        # Stream response
        response_container = st.empty()
        streamed_text = ""
        for word in answer.split():
            streamed_text += word + " "
            response_container.write(streamed_text)
            # Simulate streaming (adjust for real streaming with Google API if supported)
        
        # Add to chat history
        st.session_state.chat_history.append(("Human", query))
        st.session_state.chat_history.append(("AI", answer))
        
        # Display sources
        with st.expander("Source Documents"):
            for i, source in enumerate(sources, 1):
                st.write(f"**Source {i}**: {source}")

# Display chat history
if st.session_state.chat_history:
    st.subheader("Conversation History")
    for role, message in st.session_state.chat_history:
        if role == "Human":
            st.write(f"**You**: {message}")
        else:
            st.write(f"**AI**: {message}")