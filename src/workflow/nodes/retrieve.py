"""
Adaptive Source Router (MCP-free version, with integrated retrieval)
-------------------------------------------------------------------
Decides whether to route a query to Vector DB or Web Search based on query type
and analyzer output. If routed to vector, automatically retrieves relevant documents.
"""

from typing import Literal, Dict, Any, Set
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import json, time, os, re, sys

# --- Path setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.models.model import get_llm_model
from data.ingestion import get_retriever  # ✅ import retriever

LOG_PATH = os.environ.get("ROUTER_LOG", "logs/model_selection_log.txt")

# ---------------- STRUCTURED ROUTER ---------------- #
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Choose 'vectorstore' for AI/LLM topics, otherwise 'websearch'.",
    )

llm = get_llm_model()
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert router deciding whether a question should be answered
using an internal vectorstore or via live web search.

The vectorstore ONLY contains documents about:
- AI agents and autonomous LLM systems
- prompt engineering and instruction tuning
- adversarial attacks and LLM safety topics

If the user's question is about general knowledge, or anything not listed above,
route it to websearch.

Be strict:
- Only AI-related questions → vectorstore
- Everything else → websearch
"""

route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
question_router = route_prompt | structured_llm_router


# ---------------- FALLBACK ROUTE ---------------- #
def _fallback_route(question: str, needs_more_detail: bool, tools=None) -> Dict[str, str]:
    """Heuristic fallback when LLM fails or unavailable."""
    q = question.lower()

    if any(k in q for k in ["latest", "today", "current", "live", "now"]):
        return {"selected_source": "web", "route_reason": "Time-sensitive query detected."}
    if needs_more_detail:
        return {"selected_source": "web", "route_reason": "Query lacks sufficient detail — broad web search preferred."}
    return {"selected_source": "vector", "route_reason": "Well-defined query; local context sufficient."}


# ---------------- ROUTER ENTRY FUNCTION ---------------- #
def run(state: Dict[str, Any]) -> Dict[str, Any]:
    """Hybrid router combining structured LLM routing with adaptive heuristics."""
    question = state.get("question", "").strip()
    needs_more_detail = state.get("needs_more_detail", False)
    tools = state.get("available_tools", [])

    if not question:
        state["selected_source"] = "vector"
        state["route_reason"] = "No question provided."
        return state

    # Step 1: Run structured router
    try:
        structured_result = question_router.invoke({"question": question})
        datasource = getattr(structured_result, "datasource", "websearch")
    except Exception as e:
        print(f"[router] Structured router failed: {e}")
        datasource = "websearch"

    # Step 2: Adaptive enhancement
    if datasource == "vectorstore":
        route = _fallback_route(question, needs_more_detail)
        route["selected_source"] = "vector"
        route["route_reason"] = "AI-related topic routed to internal vector DB."
    elif datasource == "websearch":
        route = _fallback_route(question, needs_more_detail)
        route["selected_source"] = "web"
        route["route_reason"] = route.get("route_reason", "General topic routed to web search.")
    else:
        route = _fallback_route(question, needs_more_detail)

    # Step 3: Document Retrieval (if vector)
    if route["selected_source"] == "vector":
        try:
            retriever = get_retriever()
            raw_docs = retriever.invoke(question)
            print(f"[router] --- Retrieved {len(raw_docs)} docs from vectorstore ---")

            # Filter duplicates and very short content
            seen_hashes = set()
            filtered_docs = []
            unique_sources: Set[str] = set()

            for doc in raw_docs:
                text = doc.page_content.strip()
                if len(text) < 100:
                    continue
                h = hash(text)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    filtered_docs.append(doc)
                    unique_sources.add(doc.metadata.get("source", "Unknown Source"))

            if not filtered_docs:
                print("[router] ⚠️ No valid docs found — switching to web.")
                route["selected_source"] = "web"
                route["route_reason"] = "Vectorstore returned no relevant documents."
            else:
                state["documents"] = filtered_docs
                state["unique_sources"] = unique_sources
                print(f"[router] ✅ {len(filtered_docs)} unique docs kept for generation.")

        except Exception as e:
            print(f"[router] ⚠️ Retrieval error: {e}")
            route["selected_source"] = "web"
            route["route_reason"] = "Vectorstore retrieval failed — fallback to web search."

    # Step 4: Update state
    state["selected_source"] = route["selected_source"]
    state["route_reason"] = route["route_reason"]
    state["route_metadata"] = {
        "method": "HybridRouter (Structured + Adaptive + Retrieval)",
        "timestamp": time.time(),
    }

    # Step 5: Logging
    try:
        os.makedirs("logs", exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "question": question,
                "needs_more_detail": needs_more_detail,
                "selected_source": route["selected_source"],
                "reason": route["route_reason"],
                "docs_retrieved": len(state.get("documents", [])),
            }) + "\n")
    except Exception:
        pass

    print(f"[router] ✅ Routed to: {route['selected_source']} ({route['route_reason']})")
    return state
