# RAG Chatbot on PDF Documents

Welcome to the **RAG Chatbot** project — an AI-powered conversational assistant designed to answer questions by retrieving knowledge from your PDF documents in real-time! Built with modern NLP tools, this chatbot leverages Retrieval-Augmented Generation (RAG) to combine semantic search over document chunks with a powerful language model generating natural, grounded answers.

## 🚀 Project Overview

This project showcases an end-to-end pipeline to:

- Ingest and preprocess long-form PDF documents (e.g., Terms & Conditions, Privacy Policies).
- Chunk text into manageable pieces and convert them into semantic embeddings.
- Store embeddings in a vector database for fast similarity search.
- Use a prompt-tuned, open-source large language model (LLM) to generate answers augmented with retrieved context.
- Serve a friendly Streamlit interface allowing users to chat naturally with the document, getting **streaming, interactive responses** and visibility into the source text.

The pipeline uses Google’s generative AI embeddings and chat models, combined with the Chroma vector database, demonstrating modern best practices for building Retrieval-Augmented Generation chatbot solutions.


## 🛠️ Architecture & Workflow

### 1. Document Preparation

- Load and clean the PDF document using `PyPDFLoader`.
- Chunk text into ~200-word overlapping segments, preserving sentence boundaries.
- Generate semantic embeddings for each chunk using Google’s `embedding-001` model.
- Store chunks and embeddings in Chroma vector DB for efficient retrieval.

### 2. RAG Pipeline

- Initialize a retriever over the vector DB to fetch top-k relevant chunks.
- Use Google’s `gemini-pro` chat model as the generator, prompted with retrieved context plus user queries.
- Keep conversation history in memory to handle multi-turn dialogue.
- Return generated answers with citations to source chunks.

### 3. Streamlit Chat Interface

- Input field for users to submit questions about the document.
- Real-time streaming of responses (token by token simulated).
- Expandable view to show source document excerpts for transparency.
- Sidebar displays active model, embedding type, chunk count, and API key input.
- Clear chat functionality to reset conversation.

## ⚙️ Setup & Run Instructions

### Prerequisites

- Python 3.8+
- Google Cloud API key with Generative AI access.

### Installation

1. Clone the repo and navigate into it.

2. Install dependencies:

pip install -r requirements.txt


3. Place your PDF document inside the `/data` folder, e.g., `data/input.pdf`.

4. Set your Google API key (via environment variable or app sidebar).

### Run Preprocessing & Indexing

python src/preprocess.py


This will chunk your PDF, embed, and build the Chroma vector database under `/vectordb`.

### Launch the Chatbot (Streamlit)

streamlit run app.py


- Enter your Google API key in the sidebar.
- Ask questions related to your document.
- Watch answers stream in with source context for verification.

## 🔍 Model & Embedding Choices

- **Embedding model:** `embedding-001` by Google Generative AI — lightweight, powerful semantic vectors.
- **Language model:** `gemini-pro` chat model via Google Generative AI — instruction-tuned for helpful, grounded responses.
- **Vector DB:** Chroma — optimized for fast similarity search and persistence.

## 💬 Sample Queries & Outputs

Try asking:

- "What are the main privacy rights stated?"
- "How does the document define user data handling?"
- "Are there any opt-out options mentioned?"

Answers provide clear, concise info sourced directly from document chunks shown in the UI.

## ⚠️ Notes & Considerations

- Streaming responses are simulated token-by-token in the demo; real streaming depends on API support.
- Some hallucinations or incomplete answers may occur; always verify with source snippets provided.
- Performance depends on vector DB size and model response times.
- Proper API key and quota required to use Google Generative AI models.

## 🎥 Demo
Link: https://youtu.be/Cg_cwOClUBE

## 🙏 Credits & References

- [LangChain](https://langchain.com/) for modular pipelines.
- [Chroma](https://www.trychroma.com/) for vector storage.
- [Google Generative AI](https://developers.generativeai.google) APIs for embedding and generation.


