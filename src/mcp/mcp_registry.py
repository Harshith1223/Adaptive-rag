# src/mcp/mcp_registry.py
import asyncio
import logging
from typing import Dict, Any

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ✅ Create memory object stream compatibility layer
# -----------------------------------------------------------------------------
try:
    # src/mcp/mcp_registry.py

    # -----------------------------------------------------------------------------
    # ✅ Import ClientSession safely
    # -----------------------------------------------------------------------------
    try:
        from mcp.client.session import ClientSession  # Standard location for new SDK
    except Exception:
        try:
            from mcp.client import ClientSession  # Legacy location
        except Exception as e:
            raise ImportError(
                "❌ ClientSession class not found. Please ensure 'mcp' SDK is installed and up to date."
            ) from e

    # -----------------------------------------------------------------------------
    # ✅ Create memory object stream compatibility layer
    # -----------------------------------------------------------------------------
    try:
        from anyio.streams.memory import create_memory_object_stream  # AnyIO ≥4.3
    except Exception:
        # For older AnyIO builds that only expose MemoryObjectSendStream/ReceiveStream
        from anyio.streams.memory import (
            MemoryObjectSendStream,
            MemoryObjectReceiveStream,
            MemoryObjectStreamState,
        )

        def create_memory_object_stream(max_buffer_size: int = 0):
            """
            Universal fallback for AnyIO memory streams.
            Creates connected send/receive streams that share one internal queue.
            """
            state = MemoryObjectStreamState(max_buffer_size if max_buffer_size else math.inf)
            send_stream = MemoryObjectSendStream(_state=state)
            recv_stream = MemoryObjectReceiveStream(_state=state)
            return send_stream, recv_stream

    # -----------------------------------------------------------------------------
    # ✅ Imports after stream helpers
    # -----------------------------------------------------------------------------

    logger = logging.getLogger(__name__)

except Exception:
    # For older AnyIO builds that only expose MemoryObjectSendStream/ReceiveStream
    from anyio.streams.memory import (
        MemoryObjectSendStream,
        MemoryObjectReceiveStream,
        MemoryObjectStreamState,
    )
    import math

    def create_memory_object_stream(max_buffer_size: int = 0):
        """
        Universal fallback for AnyIO memory streams.
        Creates connected send/receive streams that share one internal queue.
        """
        state = MemoryObjectStreamState(max_buffer_size if max_buffer_size else math.inf)
        send_stream = MemoryObjectSendStream(_state=state)
        recv_stream = MemoryObjectReceiveStream(_state=state)
        return send_stream, recv_stream

# -----------------------------------------------------------------------------
# ✅ Imports after stream helpers
# -----------------------------------------------------------------------------
from mcp.types import ListToolsResult, Tool
import asyncio
import logging
from typing import Dict, Any
import math

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# ✅ MCP Registry Definition
# -----------------------------------------------------------------------------
class MCPRegistry:
    """
    Central registry for in-memory MCP client-server pairs.
    Handles:
      - Server registration
      - Tool discovery
      - Dynamic tool invocation
    """

    def __init__(self):
        self._clients: Dict[str, ClientSession] = {}
        self._tools: Dict[str, Tool] = {}

    # -------------------------------------------------------------------------
    async def register_server(self, name: str, server_instance):
        """
        Register an in-process MCP server and connect to it.

        Two memory stream pairs are created for bidirectional communication:
          - client_send → server_recv
          - server_send → client_recv

        Supports:
          - server.start(server_recv, server_send)
          - server.start() returning (recv_for_client, send_to_server)
        """
        try:
            # Create two stream pairs (bidirectional channel)
            client_send, server_recv = create_memory_object_stream()
            server_send, client_recv = create_memory_object_stream()

            start = getattr(server_instance, "start", None)
            if start is None:
                raise RuntimeError("Server instance has no start() method")

            # Attempt to start server with or without streams
            start_result = None
            try:
                start_result = start(server_recv, server_send)
            except TypeError:
                logger.debug(f"ℹ️ Server '{name}' start() takes no arguments.")
                start_result = start()

            # Await coroutine if needed
            if asyncio.iscoroutine(start_result):
                start_result = await start_result

            # If the server returns custom streams, use them for the client
            if isinstance(start_result, tuple) and len(start_result) == 2:
                recv_for_client, send_to_server = start_result
                client_recv = recv_for_client
                client_send = send_to_server
                logger.debug(f"ℹ️ Server '{name}' provided its own stream pair.")

            # Initialize the client session
            session = ClientSession(client_recv, client_send)
            logger.info(f"✅ Connected to in-memory MCP server '{name}'")
            # Force a tool listing so the client actually sends a request
            # 🔄 Force the client to list tools immediately after connecting
            try:
                logger.info(f"🔄 Requesting tool list from '{name}'...")
                session = ClientSession(client_recv, client_send)
                await session.initialize()
                result: ListToolsResult = await session.list_tools()
                logger.debug(f"📥 Received ListToolsResult: {result}")
                logger.info(f"📦 Received {len(result.tools or [])} tools from '{name}'.")
                for tool in result.tools or []:
                    qualified = f"{name}:{tool.name}"
                    self._tools[qualified] = tool
                    logger.info(f"🧩 Registered tool: {qualified}")
            except Exception as e:
                logger.error(f"❌ Tool listing failed for '{name}': {e}")



            # # Retrieve and register tools
            # result: ListToolsResult = await session.list_tools()
            for tool in result.tools or []:
                qualified = f"{name}:{tool.name}"
                self._tools[qualified] = tool
                logger.info(f"🧩 Registered tool: {qualified}")

            # Save the session reference
            self._clients[name] = session

        except Exception as e:
            logger.error(f"❌ Failed to register MCP server '{name}': {e}")

    # -------------------------------------------------------------------------
    async def call_tool(self, qualified_name: str, args: Dict[str, Any]):
        """
        Call a registered MCP tool.
        Example: qualified_name='web_search:search'
        """
        try:
            if ":" not in qualified_name:
                raise ValueError("qualified_name must be in format 'server:tool'")

            server_name, tool_name = qualified_name.split(":", 1)
            session = self._clients.get(server_name)
            if session is None:
                raise KeyError(f"No registered server named '{server_name}'")

            result = await session.call_tool(tool_name, args)
            return getattr(result, "content", result)
        except Exception as e:
            logger.error(f"❌ Error calling tool '{qualified_name}': {e}")
            return None


# -----------------------------------------------------------------------------
# ✅ Singleton instance (must be at bottom)
# -----------------------------------------------------------------------------
mcp_registry = MCPRegistry()
