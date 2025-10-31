"""
🌐 MCP Web Search Server — Fully compliant with MCP Spec 2025-06-18
Implements the "tools" capability per:
https://modelcontextprotocol.io/specification/2025-06-18/server/tools
"""

import asyncio
import logging
from typing import List
from mcp import Implementation
from mcp.server.session import ServerSession
from mcp.types import (
    Tool,
    ToolsCapability,
    ServerCapabilities,
    InitializeResult,
    CallToolResult,
    ListToolsResult,
    TextContent,
)

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WebSearchServer:
    """In-memory MCP server exposing a 'search' tool compliant with the 2025-06-18 spec."""

    def __init__(self):
        # Define available tools (with proper JSON schema)
        self.tools: List[Tool] = [
            Tool(
                name="search",
                description="Simulates a web search for the given query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            )
        ]

        # ✅ Capabilities as per 2025 spec
        self.capabilities = ServerCapabilities(tools=ToolsCapability())

    # -----------------------------------------------------------------
    async def start(self, server_recv, server_send):
        """Start the MCP server using in-memory streams."""
        try:
            server = ServerSession(
                read_stream=server_recv,
                write_stream=server_send,
                init_options=self.capabilities,
            )

            logger.info("✅ MCP WebSearch server started (2025 spec).")

            # ==========================================================
            # HANDLERS
            # ==========================================================
            async def handle_initialize(message):
                """Handle 'initialize' request."""
                logger.info("⚙️ Received initialize request — sending capabilities.")
                result = InitializeResult(
                    protocolVersion="2025-06-18",
                    serverInfo=Implementation(
                        name="web_search_server",
                        version="1.0.0",
                    ),
                    capabilities=self.capabilities,
                )
                await message.respond(result)
                logger.info("📤 InitializeResult sent (2025 spec).")

            async def handle_list_tools(message):
                """Handle 'tools/list' request."""
                logger.info("📦 Received tools/list request.")
                result = ListToolsResult(tools=self.tools)
                await message.respond(result)
                logger.info("📤 ListToolsResult sent.")

            async def handle_call_tool(message):
                """Handle 'tools/call' request."""
                logger.info("🔧 Received tools/call request.")
                args = getattr(message, "params", {}).arguments if hasattr(message, "params") else {}

                query = args.get("query", "")
                max_results = args.get("max_results", 2)

                # Build proper MCP TextContent result
                contents = [
                    TextContent(
                        type="text",
                        text=f"🔍 Result {i+1} for '{query}': https://example.com/{i+1}"
                    )
                    for i in range(max_results)
                ]

                result = CallToolResult(content=contents)
                await message.respond(result)
                logger.info(f"📤 CallToolResult sent for query='{query}'.")

            # ==========================================================
            # ROUTER
            # ==========================================================
            async def on_request(message):
                method = getattr(getattr(message, "method", None), "lower", lambda: None)()
                if not method:
                    logger.info("🛰️ Control or handshake message with no method.")
                    return

                handlers = {
                    "initialize": handle_initialize,
                    "tools/list": handle_list_tools,
                    "tools/call": handle_call_tool,
                }

                if method in handlers:
                    await handlers[method](message)
                else:
                    logger.warning(f"⚠️ Unknown request method: {method}")

            # ==========================================================
            # MAIN LOOP
            # ==========================================================
            server._received_request = on_request
            asyncio.create_task(server._receive_loop())

        except Exception as e:
            logger.error(f"❌ WebSearchServer failed to start: {e}")
