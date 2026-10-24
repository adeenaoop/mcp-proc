"""Smoke test for the §4 tool server. No LLM, no key.

Starts the server over real MCP stdio, then checks for one banking task:
  1. list_tools returns the same tools, in the same order, with the same schemas
  2. replaying the task's ground_truth() calls gives the same output text as
     running them directly in Python
  3. the server's final environment equals the direct-mode final environment
  4. AgentDojo's utility() check passes on the server's environments

Run from ~/fyp:  pytest tests/test_server_smoke.py -v
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

ROOT = Path(__file__).resolve().parent.parent
BENCH, SUITE, TASK = "v1.2.2", "banking", "user_task_3"


def run_direct(suite, task):
    env = task.init_environment(suite.load_and_inject_default_environment({}))
    calls = task.ground_truth(env.model_copy(deep=True))
    runtime = FunctionsRuntime(suite.tools)
    outputs = []
    for c in calls:
        result, error = runtime.run_function(env, c.function, dict(c.args))
        outputs.append(error if error is not None else tool_result_to_str(result))
    return calls, outputs, env


async def run_server(calls, ep_dir):
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "server" / "agentdojo_mcp_server.py"),
              "--suite", SUITE, "--bench-version", BENCH, "--user-task", TASK,
              "--env-pre", str(ep_dir / "env_pre.json"),
              "--env-out", str(ep_dir / "env_post.json")],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = (await session.list_tools()).tools
            outputs = []
            for c in calls:
                r = await session.call_tool(c.function, dict(c.args))
                outputs.append("".join(x.text for x in r.content if x.type == "text"))
    return listed, outputs


def test_server_matches_direct(tmp_path):
    suite = get_suite(BENCH, SUITE)
    task = suite.user_tasks[TASK]
    calls, direct_out, direct_env = run_direct(suite, task)

    listed, server_out = asyncio.run(run_server(calls, tmp_path))

    # 1. same tools, same order, same schemas
    assert [t.name for t in listed] == [f.name for f in suite.tools]
    for t, f in zip(listed, suite.tools):
        assert t.description == f.description
        assert t.input_schema == f.parameters.model_json_schema()

    # 2. same output text, call by call
    assert server_out == direct_out

    # 3. same final environment (written when the session closed)
    post_json = (tmp_path / "env_post.json").read_text()
    assert post_json == direct_env.model_dump_json()

    # 4. AgentDojo's own scoring passes on the server's environments
    EnvType = suite.environment_type
    pre = EnvType.model_validate_json((tmp_path / "env_pre.json").read_text())
    post = EnvType.model_validate_json(post_json)
    assert task.utility(task.GROUND_TRUTH_OUTPUT, pre, post)
