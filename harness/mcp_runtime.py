"""MCPRuntime: AgentDojo runtime whose tools run on the MCP server (guide §5.3).

Subclasses FunctionsRuntime with the SAME suite.tools, so the tool schemas the
LLM sees come from identical Function objects in direct and MCP mode. That
keeps the LLM's prompt byte-identical between the two modes.

Only run_function changes: instead of calling the Python function locally, it
sends the call through the bridge. The env argument is ignored, because the
real environment lives in the server process.

Tested with AgentDojo 0.1.35.
"""
from agentdojo.functions_runtime import FunctionsRuntime


class MCPRuntime(FunctionsRuntime):
    def __init__(self, tools, bridge):
        super().__init__(tools)
        self.bridge = bridge

    def run_function(self, env, function, kwargs, raise_on_error=False):
        text, is_error = self.bridge.call(function, kwargs)
        if is_error:
            if raise_on_error:
                raise RuntimeError(text)
            return "", text          # same (result, error) shape as FunctionsRuntime
        return text, None
