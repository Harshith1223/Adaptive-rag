import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from src.models.model import get_embed_model

load_dotenv()

def create_vectorstore():
    """Create or load vector store for document retrieval."""
    chroma_path = "./chroma_langchain_db"
    
    if os.path.exists(chroma_path):
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=chroma_path,
            embedding_function=get_embed_model(),
            collection_name="rag-chroma",
        )
        return vectorstore.as_retriever()
    
    print("Creating new vector store...")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=750, 
        chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=get_embed_model(),
        persist_directory=chroma_path,
    )
    
    print("Vector store created!")
    return vectorstore.as_retriever()

def get_retriever():
    """Lazily get the retriever (initializes vector store only when called)."""
    return create_vectorstore()
