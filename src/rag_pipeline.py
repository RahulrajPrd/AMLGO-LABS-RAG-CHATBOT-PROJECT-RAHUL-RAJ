from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

def create_rag_pipeline(vector_dir="vectordb"):
    # Initialize embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=vector_dir, embedding_function=embeddings, collection_name="pdf_collection")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

    # Conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # RAG pipeline
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return rag_chain

def query_rag(rag_chain, question):
    result = rag_chain({"question": question})
    return {
        "answer": result["answer"],
        "source_documents": [doc.page_content for doc in result["source_documents"]]
    }