from typing import Any, List, Set, Dict, TypedDict
from langchain_core.messages import BaseMessage

class GraphState(TypedDict, total=False):
    question: str
    user_query: str
    generation: str
    web_search: bool
    documents: List[BaseMessage]
    unique_sources: Set[str]
    web_sources: List[str]
    retry_count: int
    messages: List[BaseMessage]
    selected_source: str
    route_reason: str
    rewrite_metadata: Dict[str, Any]