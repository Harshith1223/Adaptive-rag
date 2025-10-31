# src/agents/host_agent_executor.py
import asyncio
import logging
from src.mcp.mcp_registry import mcp_registry
from src.mcp.servers.web_search_server import WebSearchServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HostAgentExecutor:
    async def main(self):
        server = WebSearchServer()
        await mcp_registry.register_server("web_search", server)

        result = await mcp_registry.call_tool("web_search:search", {"query": "Adaptive RAG system"})
        print("Tool execution result:", result)

if __name__ == "__main__":
    asyncio.run(HostAgentExecutor().main())
