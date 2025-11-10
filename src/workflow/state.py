from typing import Any, List, Set, Dict, TypedDict, Optional
from langchain_core.messages import BaseMessage

class GraphState(TypedDict, total=False):
    """
    Shared state object passed between workflow nodes.
    Contains query text, generation results, metadata, metrics, and routing details.
    """

    # --- Core workflow keys ---
    question: str
    generation: str
    rewritten_query: Optional[str]
    web_search: bool
    documents: List[BaseMessage]
    unique_sources: Set[str]
    web_sources: List[str]
    retry_count: int
    messages: List[BaseMessage]

    # --- Routing & metadata ---
    selected_source: str
    route_reason: str
    route_metadata: Dict[str, Any]
    rewrite_metadata: Dict[str, Any]

    # --- Retrieval & grading metrics ---
    factual_consistency: float         # 1.0 = factually correct, 0.0 = hallucinated
    answer_relevance: float            # 1.0 = relevant, 0.0 = irrelevant
    context_relevance: float           # cosine similarity mean of query-docs
    context_precision: float           # fraction of relevant docs (> threshold)
    confidence: float                  # weighted composite metric

    # --- Additional internal metadata ---
    retriever_type: Optional[str]      # e.g., "chroma", "faiss", "hybrid"
    docs_used: Optional[int]           # count of documents used
    needs_more_detail: Optional[bool]  # analyzer flag for query clarity
    selected_model: Optional[str]      # which LLM was used for generation
    timestamp: Optional[float]         # time of last state update
