import os
import hashlib
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from src.models.model import get_embed_model

load_dotenv()
CHROMA_PATH = "./chroma_langchain_db"
COLLECTION_NAME = "rag-chroma"


def _seed_vectorstore_if_missing() -> Chroma:
    """Create the base vector store from seed URLs when no local DB exists."""
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
        collection_name=COLLECTION_NAME,
        embedding=get_embed_model(),
        persist_directory=CHROMA_PATH,
    )
    print("Vector store created!")
    return vectorstore


def get_vectorstore() -> Chroma:
    """Create or load the persistent Chroma vector store."""
    if os.path.exists(CHROMA_PATH):
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embed_model(),
            collection_name=COLLECTION_NAME,
        )
    return _seed_vectorstore_if_missing()

def create_vectorstore():
    """Create or load vector store for document retrieval."""
    return get_vectorstore().as_retriever()


def add_web_documents_to_vectorstore(documents: list[Document]) -> int:
    """
    Add web-search documents to the same vector DB so future queries can retrieve them.
    Returns number of chunks added.
    """
    if not documents:
        return 0

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=750,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata = chunk.metadata or {}
        chunk.metadata["ingested_via"] = "web_search"

    ids = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown-source")
        raw = f"{source}|{chunk.page_content}"
        ids.append(hashlib.sha1(raw.encode("utf-8")).hexdigest())

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks, ids=ids)
    return len(chunks)

def get_retriever():
    """Lazily get the retriever (initializes vector store only when called)."""
    return create_vectorstore()
