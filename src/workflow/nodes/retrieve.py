from typing import Any, Dict, Set
from src.workflow.state import GraphState
from data.ingestion import get_retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve documents from vector store."""
    print("---RETRIEVE---")
    question = state["question"]
    
    # Get raw documents from retriever
    raw_documents = get_retriever().invoke(question)
    
    # Deduplicate by content hash, allow multiple chunks per source
    unique_sources: Set[str] = state.get("unique_sources", set())
    filtered_docs = []
    seen_hashes = set()
    
    for doc in raw_documents:
        source = doc.metadata.get("source", "Unknown Source")
        content = doc.page_content.strip()
        
        # Skip empty or very short content
        if len(content) < 100 or not content:
            print(f"---RETRIEVE: Skipping short/empty content from {source}---")
            continue
        
        # Deduplicate by content hash
        content_hash = hash(content)
        if content_hash not in seen_hashes:
            filtered_docs.append(doc)
            seen_hashes.add(content_hash)
            unique_sources.add(source)
            print(f"---RETRIEVE: Kept document from {source}---")
        else:
            print(f"---RETRIEVE: Skipping duplicate content from {source}---")
    
    print(f"---RETRIEVE: {len(filtered_docs)} unique documents retrieved---")
    return {
        "documents": filtered_docs,
        "question": question,
        "unique_sources": unique_sources
    }