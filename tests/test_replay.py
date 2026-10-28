"""X1: replay equivalence, no LLM (guide §7.1). AgentDojo 0.1.35, mcp 2.3.0.

For every banking USER task and every banking INJECTION task, runs the task's
ground_truth() tool calls twice:
  direct side: local FunctionsRuntime, outputs formatted with tool_result_to_str
  MCP side:    MCPBridge -> (mcpfw, if enabled) -> tool server
and asserts:
  1. output strings are equal, call by call, as bytes, AND the error status
     matches (direct: error is not None; MCP: is_error)
  2. post_direct.model_dump_json() == contents of the server's env_post.json
  3. the task's check passes on BOTH sides
     (user tasks: utility; injection tasks: security)

Scoring copies AgentDojo's _check_task_result exactly: use
utility_from_traces / security_from_traces if it returns non-None, otherwise
utility / security(model_output, pre, post).

X1 is the PASS-THROUGH baseline: with X1_VIA_PROXY=1, mcpfw must only forward
and log. Once mcpfw enforces security rules (later weeks), denied calls are
expected to differ; test those in separate denial tests, not here.

Run from ~/fyp:
    pytest tests/test_replay.py -v                   # harness <-> server
    X1_VIA_PROXY=1 pytest tests/test_replay.py -v    # harness <-> mcpfw <-> server (after §8)
"""
import os
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from bridge import MCPBridge  # noqa: E402

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

BENCH, SUITE = "v1.2.2", "banking"
SERVER = ROOT / "server" / "agentdojo_mcp_server.py"
PROXY = ROOT / "mcpfw" / "proxy.py"
VIA_PROXY = os.environ.get("X1_VIA_PROXY") == "1"
# Extra mcpfw options, e.g. MCPFW_EXTRA_ARGS="--policy policies/banking.json"
# (used to rerun X1/X2/X3 with a policy loaded in audit mode)
MCPFW_EXTRA_ARGS = shlex.split(os.environ.get("MCPFW_EXTRA_ARGS", ""))

suite = get_suite(BENCH, SUITE)
EnvType = suite.environment_type


def numeric(ids):
    return sorted(ids, key=lambda k: int(k.rsplit("_", 1)[-1]))


# ---------- setup shared by both sides (§4.3 steps 1-4) ----------
def initial_env(user_task=None):
    env = suite.load_and_inject_default_environment({})
    if user_task is not None:                       # AgentDojo only inits for user tasks
        env = user_task.init_environment(env)
    return env


# ---------- scoring: copied from AgentDojo's _check_task_result ----------
def check(task, is_user_task, pre, post, calls):
    output = task.GROUND_TRUTH_OUTPUT
    if is_user_task:
        r = task.utility_from_traces(output, pre, post, calls)
        return r if r is not None else task.utility(output, pre, post)
    r = task.security_from_traces(output, pre, post, calls)
    return r if r is not None else task.security(output, pre, post)


# ---------- direct side ----------
def run_direct(task, is_user_task):
    env = initial_env(task if is_user_task else None)
    pre = env.model_copy(deep=True)
    calls = task.ground_truth(pre.model_copy(deep=True))
    runtime = FunctionsRuntime(suite.tools)
    outputs = []
    for c in calls:
        result, error = runtime.run_function(env, c.function, dict(c.args))
        if error is not None:
            outputs.append((error, True))
        else:
            outputs.append((tool_result_to_str(result), False))
    return calls, outputs, pre, env


# ---------- MCP side ----------
def mcp_command(ep_dir, user_task_id):
    server_args = [str(SERVER), "--suite", SUITE, "--bench-version", BENCH,
                   "--env-pre", str(ep_dir / "env_pre.json"),
                   "--env-out", str(ep_dir / "env_post.json")]
    if user_task_id:
        server_args += ["--user-task", user_task_id]
    if not VIA_PROXY:
        return sys.executable, server_args
    # §8.1: mcpfw --log <trace> -- <server command and args>
    return sys.executable, [str(PROXY), "--log", str(ep_dir / "trace.jsonl"), *MCPFW_EXTRA_ARGS,
                            "--", sys.executable, *server_args]


def run_mcp(calls, ep_dir, user_task_id):
    command, args = mcp_command(ep_dir, user_task_id)
    bridge = MCPBridge(command, args, expected_tools={f.name for f in suite.tools})
    try:
        outputs = []
        for c in calls:
            text, is_error = bridge.call(c.function, dict(c.args))
            outputs.append((text, is_error))
    finally:
        bridge.close()                               # server writes env_post.json
    post_path = ep_dir / "env_post.json"
    assert post_path.exists(), "server did not write env_post.json"
    pre = EnvType.model_validate_json((ep_dir / "env_pre.json").read_text())
    return outputs, pre, post_path.read_text()


def first_diff(a, b):
    """Describe where two strings first differ, for readable failures."""
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"first difference at char {i}: direct={a[i:i+40]!r} mcp={b[i:i+40]!r} (lengths {len(a)} vs {len(b)})"


# ---------- the test ----------
def replay(task, is_user_task, ep_dir):
    calls, direct_out, direct_pre, direct_post = run_direct(task, is_user_task)
    mcp_out, mcp_pre, mcp_post_json = run_mcp(calls, ep_dir, task.ID if is_user_task else None)

    # 1. outputs equal, call by call, as bytes; error status equal too
    assert len(mcp_out) == len(direct_out) == len(calls)
    for i, ((d_text, d_err), (m_text, m_err)) in enumerate(zip(direct_out, mcp_out)):
        where = f"call {i} ({calls[i].function})"
        assert d_err == m_err, f"{where} error status differs: direct={d_err} mcp={m_err}"
        assert d_text.encode() == m_text.encode(), f"{where} output differs; {first_diff(d_text, m_text)}"

    # 2. identical final environments (strict: serialized JSON strings)
    direct_post_json = direct_post.model_dump_json()
    assert mcp_post_json == direct_post_json, \
        f"post-environment differs; {first_diff(direct_post_json, mcp_post_json)}"
    # (and identical starting environments, so both sides began from the same state)
    d_pre, m_pre = direct_pre.model_dump_json(), mcp_pre.model_dump_json()
    assert m_pre == d_pre, f"pre-environment differs; {first_diff(d_pre, m_pre)}"

    # 3. the task's check passes on both sides
    mcp_post = EnvType.model_validate_json(mcp_post_json)
    name = "utility" if is_user_task else "security"
    assert check(task, is_user_task, direct_pre, direct_post, calls), f"direct {name} failed"
    assert check(task, is_user_task, mcp_pre, mcp_post, calls), f"MCP {name} failed"


@pytest.mark.parametrize("task_id", numeric(suite.user_tasks))
def test_user_task(task_id, tmp_path):
    replay(suite.user_tasks[task_id], True, tmp_path)


@pytest.mark.parametrize("task_id", numeric(suite.injection_tasks))
def test_injection_task(task_id, tmp_path):
    replay(suite.injection_tasks[task_id], False, tmp_path)


# ---------- error path ----------
# Ground-truth calls never fail (0 errors in v1.2.2 banking), so the error
# check above only ever compares "no error" with "no error". These deliberate
# bad calls make X1 cover the error path too, including through mcpfw.
class _Call:
    def __init__(self, function, args):
        self.function, self.args = function, args


BAD_CALLS = [
    _Call("send_money", {"recipient": "X"}),                      # missing required args
    _Call("send_money", {"recipient": "X", "amount": "lots",       # wrong argument type
                         "subject": "s", "date": "2022-01-01"}),
    _Call("not_a_real_tool", {}),                                  # unknown tool
]


def test_error_calls(tmp_path):
    env = initial_env()
    runtime = FunctionsRuntime(suite.tools)
    direct = []
    for c in BAD_CALLS:
        result, error = runtime.run_function(env, c.function, dict(c.args))
        direct.append((error, True) if error is not None else (tool_result_to_str(result), False))
    assert all(is_err for _, is_err in direct), "every bad call should error in direct mode"

    mcp, _, _ = run_mcp(BAD_CALLS, tmp_path, None)
    assert len(mcp) == len(direct) == len(BAD_CALLS)    # zip() would silently drop extras
    # Error TEXT must match byte for byte, not just status: in MCP mode the
    # error text is what the LLM reads, so a difference would change its prompt.
    for i, ((d_text, d_err), (m_text, m_err)) in enumerate(zip(direct, mcp)):
        assert d_err == m_err, f"bad call {i} error status differs: direct={d_err} mcp={m_err}"
        assert d_text.encode() == m_text.encode(), f"bad call {i} error text differs; {first_diff(d_text, m_text)}"
