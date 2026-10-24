"""Tests for §5: MCPBridge + MCPRuntime. No LLM, no key.

For every banking user task, runs the task's ground_truth() tool calls through
AgentDojo's real ToolsExecutor twice: once with the normal FunctionsRuntime
(direct mode) and once with MCPRuntime -> MCPBridge -> tool server (MCP mode).
ToolsExecutor is the exact component an LLM agent uses, so this checks the
path the agent will take.

Asserts, per task:
  - identical tool-result messages (content text and error), call by call
  - identical final environment (direct env vs the server's env_post.json)
Plus:
  - a bad tool call gives the same error in both modes
  - the startup assertion rejects a wrong tool set
  - the default ToolsExecutor leaves an already-formatted string unchanged (§5.4)

Run from ~/fyp:  pytest tests/test_bridge.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from bridge import MCPBridge            # noqa: E402
from mcp_runtime import MCPRuntime      # noqa: E402

from agentdojo.agent_pipeline import ToolsExecutor
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import get_text_content_as_str

BENCH, SUITE = "v1.2.2", "banking"
SERVER = str(ROOT / "server" / "agentdojo_mcp_server.py")
suite = get_suite(BENCH, SUITE)
TASK_IDS = sorted(suite.user_tasks, key=lambda k: int(k.rsplit("_", 1)[-1]))


def start_bridge(ep_dir, user_task=None, expected=None):
    args = [SERVER, "--suite", SUITE, "--bench-version", BENCH,
            "--env-pre", str(ep_dir / "env_pre.json"),
            "--env-out", str(ep_dir / "env_post.json")]
    if user_task:
        args += ["--user-task", user_task]
    names = {f.name for f in suite.tools} if expected is None else expected
    return MCPBridge(sys.executable, args, expected_tools=names)


def run_executor(runtime, env, calls):
    """Feed each call to ToolsExecutor as if an LLM had requested it.
    Returns [(content_text, error), ...]."""
    executor = ToolsExecutor()
    out = []
    for i, c in enumerate(calls):
        call = FunctionCall(function=c.function, args=dict(c.args), id=f"call_{i}")
        assistant = {"role": "assistant", "content": None, "tool_calls": [call]}
        _, _, env, messages, _ = executor.query("", runtime, env, [assistant])
        msg = messages[-1]
        assert msg["role"] == "tool"
        out.append((get_text_content_as_str(msg["content"]), msg["error"]))
    return out, env


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_mcp_matches_direct(task_id, tmp_path):
    task = suite.user_tasks[task_id]

    # Direct mode
    direct_env = task.init_environment(suite.load_and_inject_default_environment({}))
    calls = task.ground_truth(direct_env.model_copy(deep=True))
    direct_out, direct_env = run_executor(FunctionsRuntime(suite.tools), direct_env, calls)

    # MCP mode (env passed to the executor is a placeholder; the real one is in the server)
    bridge = start_bridge(tmp_path, user_task=task_id)
    try:
        placeholder = suite.load_and_inject_default_environment({})
        mcp_out, _ = run_executor(MCPRuntime(suite.tools, bridge), placeholder, calls)
    finally:
        bridge.close()

    assert mcp_out == direct_out
    assert (tmp_path / "env_post.json").read_text() == direct_env.model_dump_json()


def test_error_matches_direct(tmp_path):
    bad = [FunctionCall(function="send_money", args={"recipient": "X"}, id="bad")]  # missing args
    env = suite.load_and_inject_default_environment({})
    direct_out, _ = run_executor(FunctionsRuntime(suite.tools), env, bad)

    bridge = start_bridge(tmp_path)
    try:
        mcp_out, _ = run_executor(MCPRuntime(suite.tools, bridge), env, bad)
    finally:
        bridge.close()

    assert direct_out[0][1] is not None, "direct mode should report an error"
    assert mcp_out == direct_out


def test_startup_rejects_wrong_tool_set(tmp_path):
    with pytest.raises(RuntimeError, match="Tool set mismatch"):
        start_bridge(tmp_path, expected={"not_a_real_tool"})


def test_formatter_leaves_strings_unchanged():
    """§5.4: if this holds, MCP mode can use the default ToolsExecutor()."""
    for s in ["plain", "- amount: 4.0\n  id: 1\n", "", "multi\nline: yes"]:
        assert tool_result_to_str(s) == s
