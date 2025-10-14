from typing import Any, Dict, Set
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch
from src.workflow.state import GraphState

load_dotenv()
MAX_RETRIES = 3
web_search_tool = TavilySearch(max_results=4)  # choose appropriate number

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]

    # ALWAYS start fresh for web search results
    documents = []  # DO NOT reuse state["documents"] here
    unique_sources: Set[str] = set()

    # Reset generation to avoid stale bad answers
    state["generation"] = ""

    # Increment retry_count for web search attempts
    retry_count = state.get("retry_count", 0) + 1
                    
    if retry_count > MAX_RETRIES:
        print("---MAX RETRIES EXCEEDED IN WEB SEARCH---")
        return {
            "documents": documents,
            "question": question,
            "generation": "Unable to find a reliable answer after multiple attempts. Please refine your query.",
            "web_search": False,
            "web_sources": [],
            "unique_sources": unique_sources,
            "retry_count": retry_count
        }

    results = web_search_tool.invoke({"query": question})["results"]

    # Build new documents list from web results (no mixing)
    new_sources = []
    for res in results:
        url = res.get("url", "")
        if not url or url in new_sources:
            continue
        new_sources.append(url)
        unique_sources.add(url)
        doc = Document(
            page_content=res.get("content", ""),
            metadata={"source": url, "title": res.get("title", "Web Result")}
        )
        documents.append(doc)

    print(f"---WEB SEARCH: Retrieved {len(documents)} web docs---")

    return {
        "documents": documents,
        "question": question,
        "web_search": True,
        "web_sources": new_sources,
        "unique_sources": unique_sources,
        "retry_count": retry_count
    }
