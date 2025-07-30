import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import json
from dotenv import load_dotenv

load_dotenv()

def process_pdf(pdf_path, chunk_dir="chunks", vector_dir="vectordb"):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Clean and chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, # 100-300 words
        chunk_overlap=20,
        separators=["\n\n", "\n", ".", "!", "?", ","]
    )
    chunks = text_splitter.split_documents(pages)

    # Save chunks
    os.makedirs(chunk_dir, exist_ok=True)
    with open(os.path.join(chunk_dir, "chunks.json"), "w") as f:
        json.dump([{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks], f)

    # Generate and store embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vector_dir,
        collection_name="pdf_collection"
    )
    vector_store.persist()
    print(f"Processed {len(chunks)} chunks and stored in {vector_dir}")

if __name__ == "__main__":
    process_pdf("data/input.pdf")