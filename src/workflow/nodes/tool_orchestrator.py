from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from src.workflow.state import GraphState
from data.ingestion import add_web_documents_to_vectorstore

load_dotenv()


class MCPToolOrchestrator:
    """
    Lightweight MCP-style tool orchestrator.
    Selects and executes a tool based on query intent and available tools.
    """

    def __init__(self):
        self.search_tool = None

    def _select_tool(self, question: str, available_tools: List[str]) -> str:
        q = (question or "").lower()
        tools = [t.lower() for t in (available_tools or [])]

        if any(k in q for k in ["weather", "temperature", "forecast"]) and "weather_api" in tools:
            return "weather_api"
        if any(k in q for k in ["stock", "share", "ticker", "price"]) and "stock_api" in tools:
            return "stock_api"
        if "web_search" in tools:
            return "web_search"
        return tools[0] if tools else "web_search"

    def _build_query_for_tool(self, tool_name: str, question: str) -> str:
        if tool_name == "weather_api":
            return f"Current weather details: {question}"
        if tool_name == "stock_api":
            return f"Latest stock market info: {question}"
        return question

    def run(self, question: str, available_tools: List[str]) -> Dict[str, Any]:
        selected_tool = self._select_tool(question, available_tools)
        tool_query = self._build_query_for_tool(selected_tool, question)
        if self.search_tool is None:
            self.search_tool = TavilySearch(max_results=4)
        results = self.search_tool.invoke({"query": tool_query}).get("results", [])

        documents = []
        for res in results:
            documents.append(
                Document(
                    page_content=res.get("content", ""),
                    metadata={
                        "source": res.get("url", ""),
                        "title": res.get("title", selected_tool),
                        "tool": selected_tool,
                    },
                )
            )

        cached_chunks = 0
        try:
            cached_chunks = add_web_documents_to_vectorstore(documents)
        except Exception:
            cached_chunks = 0

        return {
            "selected_tool": selected_tool,
            "tool_documents": documents,
            "web_cached_chunks": cached_chunks,
        }


tool_orchestrator = MCPToolOrchestrator()


def run(state: GraphState) -> Dict[str, Any]:
    """
    Tool execution node for queries routed to `selected_source=tool`.
    """
    question = state.get("question", "")
    available_tools = state.get("available_tools", [])
    result = tool_orchestrator.run(question, available_tools)
    documents = result.get("tool_documents", [])

    return {
        "question": question,
        "documents": documents,
        "web_search": True,
        "selected_tool": result.get("selected_tool"),
        "web_cached_chunks": result.get("web_cached_chunks", 0),
        "route_reason": f"Executed tool orchestrator via {result.get('selected_tool')}",
    }
