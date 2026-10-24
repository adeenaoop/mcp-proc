"""AgentDojo tool server over MCP stdio (guide §4), process C.

Exposes one AgentDojo suite's tools over MCP and holds the environment those
tools mutate. Tested with AgentDojo 0.1.35 and mcp 2.3.0.

Usage:
    python server/agentdojo_mcp_server.py --suite banking --bench-version v1.2.2 \
        --user-task user_task_3 --injections run/ep/injections.json \
        --env-pre run/ep/env_pre.json --env-out run/ep/env_post.json

--user-task is optional: omit it when replaying an injection task's
ground truth, because AgentDojo only calls init_environment for user tasks.

RULE (§4.6): nothing may be printed to stdout. Stdout is the MCP channel.
All logging goes to stderr via log().
"""
import argparse
import json
import sys
from pathlib import Path

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite


def log(*args):
    print("[server]", *args, file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--bench-version", required=True)
    p.add_argument("--user-task", default=None)
    p.add_argument("--injections", default=None, help="JSON file; omitted = {} (benign)")
    p.add_argument("--env-pre", required=True)
    p.add_argument("--env-out", required=True)
    return p.parse_args()


def setup(args):
    """§4.3 steps 1-7, in this exact order (mirrors run_task_with_pipeline)."""
    # 1. Load the suite at the pinned benchmark version.
    suite = get_suite(args.bench_version, args.suite)

    # 2. Read injections: {} for benign, otherwise dict[str, str].
    injections = {}
    if args.injections:
        injections = json.loads(Path(args.injections).read_text())

    # 3. Load the default environment with injections applied.
    env = suite.load_and_inject_default_environment(injections)

    # 4. Let the user task modify the environment. Skipping this silently changes results.
    if args.user_task:
        env = suite.user_tasks[args.user_task].init_environment(env)

    # 5. Save the starting environment, for scoring later.
    Path(args.env_pre).parent.mkdir(parents=True, exist_ok=True)
    Path(args.env_pre).write_text(env.model_dump_json())

    # 6. The same runtime direct mode uses, so tools behave identically.
    runtime = FunctionsRuntime(suite.tools)

    # 7. Tool table, in suite.tools order (order changes the LLM's prompt).
    tools = [
        types.Tool(
            name=f.name,
            description=f.description,
            input_schema=f.parameters.model_json_schema(),
        )
        for f in suite.tools
    ]
    log(f"suite={args.suite} bench={args.bench_version} task={args.user_task} "
        f"injections={len(injections)} tools={len(tools)}")
    return env, runtime, tools


async def serve(args):
    env, runtime, tools = setup(args)

    # §4.4 list_tools: return the table from step 7, unchanged.
    async def on_list_tools(ctx, params):
        return types.ListToolsResult(tools=tools)

    # §4.4 call_tool: run the real AgentDojo function against the env.
    async def on_call_tool(ctx, params):
        name, arguments = params.name, params.arguments or {}
        result, error = runtime.run_function(env, name, arguments)   # mutates env in place
        if error is not None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=error)], is_error=True)
        # Same formatter ToolsExecutor uses by default, so outputs match direct mode.
        text = tool_result_to_str(result)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)], is_error=False)

    server = Server("agentdojo-tools", on_list_tools=on_list_tools, on_call_tool=on_call_tool)

    try:
        async with stdio_server() as (read_stream, write_stream):
            # §4.6 safety net: stdio_server has already taken the real stdout,
            # so any stray print from here on goes to stderr instead of
            # corrupting the protocol channel.
            sys.stdout = sys.stderr
            # 8. Serve until the client closes our stdin (EOF).
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        # §4.5: always write the final environment, even after an error.
        Path(args.env_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.env_out).write_text(env.model_dump_json())
        log("wrote", args.env_out)


def main():
    anyio.run(serve, parse_args())


if __name__ == "__main__":
    main()
