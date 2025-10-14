from typing import Annotated, Dict, List, TypedDict
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: Annotated[str, ""]
    web_search: bool
    documents: List[BaseMessage]  # Adjust if documents are not BaseMessage
    unique_sources: set
    web_sources: list
    retry_count: int
    messages: Annotated[List[BaseMessage], "List of conversation messages"]