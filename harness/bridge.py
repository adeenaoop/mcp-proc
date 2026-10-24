"""MCPBridge: synchronous front end to an asynchronous MCP client (guide §5.2).

AgentDojo's ToolsExecutor calls tools synchronously. The MCP client SDK is
asynchronous. The bridge runs the MCP client on its own event loop in a
background thread and exposes a plain blocking call() method.

Tested with mcp 2.3.0.

Design note: the MCP SDK's context managers (stdio_client, ClientSession) must
be entered and exited by the same asyncio task. So one long-lived task,
_lifecycle(), opens them, waits until close() is requested, then closes them.
call() submits each tool call to the same loop as a separate short task.

Usage:
    bridge = MCPBridge(sys.executable, ["server/agentdojo_mcp_server.py", ...],
                       expected_tools={f.name for f in suite.tools})
    try:
        text, is_error = bridge.call("get_balance", {})
    finally:
        bridge.close()     # closes the server's stdin -> it writes env_post.json
"""
import asyncio
import threading

from mcp import ClientSession, StdioServerParameters, stdio_client

START_TIMEOUT_S = 60
CALL_TIMEOUT_S = 120
CLOSE_TIMEOUT_S = 30


class MCPBridge:
    def __init__(self, command, args, expected_tools=None, env=None):
        """command: the program to start (the server now; mcpfw from §8).
        args: its arguments. expected_tools: tool names the server must offer.
        env: environment variables for the child (None = inherit)."""
        self._params = StdioServerParameters(command=command, args=list(args), env=env)
        self._expected = set(expected_tools) if expected_tools is not None else None
        self.session = None
        self.tools = None            # list_tools() result, kept for checks
        self._closed = False

        # A private event loop running in a background thread.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._ready = threading.Event()
        self._startup_error = None
        self._stop = None            # asyncio.Event, created inside the loop
        self._lifecycle_future = asyncio.run_coroutine_threadsafe(self._lifecycle(), self._loop)

        # Block until the session is ready or startup failed.
        if not self._ready.wait(START_TIMEOUT_S):
            self._shutdown_loop()
            raise TimeoutError(f"MCP server did not start within {START_TIMEOUT_S}s")
        if self._startup_error is not None:
            self._shutdown_loop()
            raise self._startup_error

    async def _lifecycle(self):
        """Open transport + session, initialise, list tools, wait, then close."""
        self._stop = asyncio.Event()
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.tools = (await session.list_tools()).tools

                    # Startup assertion (§5.2): exactly the expected tool names.
                    # Recorded rather than raised here: raising inside the SDK's
                    # contexts would wrap it in an ExceptionGroup.
                    if self._expected is not None:
                        got = {t.name for t in self.tools}
                        if got != self._expected:
                            self._startup_error = RuntimeError(
                                "Tool set mismatch. Missing: "
                                f"{sorted(self._expected - got)}; unexpected: {sorted(got - self._expected)}")
                            self._ready.set()
                            return                   # contexts close cleanly

                    self.session = session
                    self._ready.set()
                    await self._stop.wait()          # stay open until close()
        except BaseException as e:
            if not self._ready.is_set():             # failed during startup
                self._startup_error = e
                self._ready.set()
                return
            raise

    def call(self, name, args):
        """Run one tool on the server. Returns (text, is_error)."""
        if self._closed or self.session is None:
            raise RuntimeError("MCPBridge is closed")
        future = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(name, dict(args)), self._loop)
        result = future.result(timeout=CALL_TIMEOUT_S)
        # Join the text of all TextContent items (mcp 2.x: snake_case is_error).
        text = "".join(c.text for c in result.content if getattr(c, "type", None) == "text")
        return text, bool(result.is_error)

    def close(self):
        """Close session and transport (the server's stdin gets EOF and the
        server writes env_post.json), then stop the loop and join the thread.
        Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._stop is not None:
                self._loop.call_soon_threadsafe(self._stop.set)
            self._lifecycle_future.result(timeout=CLOSE_TIMEOUT_S)
        finally:
            self.session = None
            self._shutdown_loop()

    def _shutdown_loop(self):
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=CLOSE_TIMEOUT_S)
        if not self._loop.is_running():
            self._loop.close()

    # Allows: with MCPBridge(...) as bridge: ...
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
