# test_mcp_registry.py
import asyncio
from src.mcp.mcp_registry import mcp_registry

async def main():
    # Example: connect to a test MCP server later (for now this will fail gracefully)
    await mcp_registry.register_server("web_search", "localhost", 8081)

    print("Tools discovered:", list(mcp_registry._tools.keys()))

    # Try to call a tool (will error until server exists)
    result = await mcp_registry.call_tool("web_search:search", {"query": "Adaptive RAG"})
    print("Result:", result)

    await mcp_registry.close_all()

if __name__ == "__main__":
    asyncio.run(main())
