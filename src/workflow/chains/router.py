# src/workflow/chains/router.py
"""
Adaptive Source Router
----------------------
Extends existing router to dynamically decide whether to route a query
to Vector DB, Web Search, or MCP Tool based on query type, analyzer output,
and available tools.

Combines:
- Your structured router (RouteQuery) for vectorstore/websearch classification
- New adaptive Source Selector logic (tool detection + analyzer integration)
"""

from typing import Literal, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import json, time, os, re, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.models.model import get_llm_model

LOG_PATH = os.environ.get("ROUTER_LOG", "logs/model_selection_log.txt")

# ---------------- STRUCTURED ROUTER (YOUR EXISTING LOGIC) ---------------- #
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

route_prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{question}")]
)

question_router = route_prompt | structured_llm_router


# ---------------- NEW ADAPTIVE SOURCE SELECTOR (STEP 3 UPGRADE) ---------------- #
def _fallback_route(question: str, needs_more_detail: bool, tools=None) -> Dict[str, str]:
    """Heuristic fallback when LLM fails or unavailable."""
    q = question.lower()
    if any(k in q for k in ["latest", "today", "current", "live", "now"]):
        return {"selected_source": "web", "route_reason": "Time-sensitive query detected."}
    if any(k in q for k in ["weather", "stock", "price", "currency"]) and tools:
        return {"selected_source": "tool", "route_reason": "MCP tool available for domain data."}
    if needs_more_detail:
        return {"selected_source": "web", "route_reason": "Query lacks sufficient detail — broad web search preferred."}
    return {"selected_source": "vector", "route_reason": "Well-defined query; local context sufficient."}


def run(state: Dict) -> Dict:
    """
    Combines structured LLM routing (AI vs general) with adaptive multi-source logic.
    Returns selected_source in: {'vector', 'web', 'tool'}
    """
    question = state.get("question") or ""
    needs_more_detail = state.get("needs_more_detail", False)
    tools = state.get("available_tools", [])

    if not question:
        state["selected_source"] = "vector"
        state["route_reason"] = "No question provided."
        return state

    # Step 1: Run your existing structured router first
    try:
        structured_result = question_router.invoke({"question": question})
        datasource = getattr(structured_result, "datasource", "websearch")
    except Exception as e:
        print(f"[router] Structured router failed: {e}")
        datasource = "websearch"

    # Step 2: Adaptive enhancement
    if datasource == "vectorstore":
        route = _fallback_route(question, needs_more_detail, tools)
        route["selected_source"] = "vector"
        route["route_reason"] = "AI-related topic routed to internal vector DB."
    elif datasource == "websearch":
        route = _fallback_route(question, needs_more_detail, tools)
        route["selected_source"] = "web"
        route["route_reason"] = route.get("route_reason", "General topic routed to web search.")
    else:
        route = _fallback_route(question, needs_more_detail, tools)

    # Step 3: MCP-aware override
    if route["selected_source"] != "tool" and tools:
        q_lower = question.lower()
        for tool_name in tools:
            if tool_name.lower() in q_lower:
                route["selected_source"] = "tool"
                route["route_reason"] = f"Direct tool match detected for '{tool_name}'."
                break

    # Step 4: Update state
    state["selected_source"] = route["selected_source"]
    state["route_reason"] = route["route_reason"]
    state["route_metadata"] = {
        "method": "HybridRouter (Structured + Adaptive)",
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
                "reason": route["route_reason"]
            }) + "\n")
    except Exception:
        pass

    print(f"[router] ✅ Routed to: {route['selected_source']} ({route['route_reason']})")
    return state
