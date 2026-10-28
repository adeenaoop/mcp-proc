"""MCP-mode episode runner (guide §6). AgentDojo 0.1.35, mcp 2.3.0.

Why it exists: run_task_with_pipeline() scores against the environment object
inside the harness, but in MCP mode the tools run in the server, so that
object never changes. This runner copies AgentDojo's episode logic exactly
and scores against the environments the server writes (env_pre/env_post).

Path of every tool call:  agent -> MCPRuntime -> MCPBridge -> mcpfw -> server
(--no-proxy skips mcpfw)

Usage (from ~/fyp, venv active):
    python harness/run_episode.py --fake-gt           # no key: fake model follows ground truth
    python harness/run_episode.py                     # real: first 5 banking tasks, 1 rep
    python harness/run_episode.py --reps 3            # X4 MCP side: 5 tasks x 3 reps

Outputs, per episode, in run/<ep_id>/  (fake runs: run/fake/<ep_id>/):
    injections.json, env_pre.json, env_post.json, trace.jsonl,
    messages.json (full conversation), result.json (§3.4 format, mode "mcp")
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(ROOT / "mcpfw"))

from agentdojo.agent_pipeline.errors import AbortAgentError  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import get_text_content_as_str  # noqa: E402

from bridge import MCPBridge  # noqa: E402
from counting_client import CountingClient  # noqa: E402
from events import verify_chain  # noqa: E402
from fake_llm import GroundTruthFakeClient  # noqa: E402
from mcp_runtime import MCPRuntime  # noqa: E402
from run_direct import build_pipeline, make_client, numeric_task_ids  # noqa: E402

SERVER = ROOT / "server" / "agentdojo_mcp_server.py"
PROXY = ROOT / "mcpfw" / "proxy.py"
ENV_POST_TIMEOUT_S = 10
MAX_ATTEMPTS = 3


# ---------- copied from agentdojo.task_suite.task_suite (0.1.35) ----------
def model_output_from_messages(messages):
    if messages[-1]["role"] != "assistant":
        raise ValueError("Last message was not an assistant message")
    return messages[-1]["content"]


def functions_stack_trace_from_messages(messages):
    calls = []
    for m in messages:
        if m["role"] == "assistant":
            calls.extend(m["tool_calls"] or [])
    return calls


def check_task_result(task, is_user_task, model_output, pre, post, traces):
    """Same rule as TaskSuite._check_task_result: *_from_traces first,
    then utility/security if that returned None."""
    text = get_text_content_as_str(model_output)
    if is_user_task:
        r = task.utility_from_traces(text, pre, post, traces)
        return r if r is not None else task.utility(text, pre, post)
    r = task.security_from_traces(text, pre, post, traces)
    return r if r is not None else task.security(text, pre, post)
# ---------------------------------------------------------------------------


def to_json(obj):
    """messages contain pydantic objects (FunctionCall); make them JSON-safe."""
    return json.dumps(obj, indent=2, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o))


def server_command(ep_dir, suite_name, bench, user_task_id, via_proxy):
    server_args = [str(SERVER), "--suite", suite_name, "--bench-version", bench,
                   "--user-task", user_task_id,
                   "--injections", str(ep_dir / "injections.json"),
                   "--env-pre", str(ep_dir / "env_pre.json"),
                   "--env-out", str(ep_dir / "env_post.json")]
    if not via_proxy:
        return sys.executable, server_args
    return sys.executable, [str(PROXY), "--log", str(ep_dir / "trace.jsonl"),
                            "--", sys.executable, *server_args]


def run_episode(suite, suite_name, bench, user_task_id, rep, client, model, out_root,
                injection_task_id=None, injections=None, via_proxy=True, force=False):
    user_task = suite.user_tasks[user_task_id]
    injection_task = suite.injection_tasks[injection_task_id] if injection_task_id else None
    injections = injections or {}

    # 1. run/<ep_id>/
    ep_id = f"{suite_name}_{user_task_id}_{injection_task_id or 'none'}_r{rep}"
    ep_dir = out_root / ep_id
    if ep_dir.exists():
        if not force:
            raise FileExistsError(f"{ep_dir} exists; use --force to replace it")
        shutil.rmtree(ep_dir)
    ep_dir.mkdir(parents=True)

    # 2. injections.json
    (ep_dir / "injections.json").write_text(json.dumps(injections))

    pipeline = build_pipeline(client, model)
    client.reset()
    t0 = time.monotonic()

    # 3. start the bridge (-> mcpfw -> server)
    command, args = server_command(ep_dir, suite_name, bench, user_task_id, via_proxy)
    bridge = MCPBridge(command, args, expected_tools={f.name for f in suite.tools})
    try:
        # 4. runtime with the SAME suite.tools as direct mode
        runtime = MCPRuntime(suite.tools, bridge)

        # 5. up to 3 attempts, AgentDojo's retry rule. One runtime and one server
        #    for all attempts, as in AgentDojo, so environment changes carry over.
        placeholder_env = suite.load_and_inject_default_environment(injections)  # ignored by MCPRuntime
        model_output, messages, attempts = None, [], 0
        for _ in range(MAX_ATTEMPTS):
            attempts += 1
            try:
                _, _, _, messages, _ = pipeline.query(user_task.PROMPT, runtime, placeholder_env)
            except AbortAgentError as e:
                messages = e.messages
            model_output = model_output_from_messages(messages)
            if model_output is not None:
                break
    finally:
        # 6. close: the server sees EOF and writes env_post.json
        bridge.close()

    post_path = ep_dir / "env_post.json"
    deadline = time.monotonic() + ENV_POST_TIMEOUT_S
    while not post_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not post_path.exists():
        raise RuntimeError(f"env_post.json missing after {ENV_POST_TIMEOUT_S}s: {post_path}")
    wall_s = round(time.monotonic() - t0, 2)

    # 7. environments as the server saw them
    EnvType = suite.environment_type
    pre = EnvType.model_validate_json((ep_dir / "env_pre.json").read_text())
    post = EnvType.model_validate_json(post_path.read_text())

    # 8. traces from the conversation
    traces = functions_stack_trace_from_messages(messages)

    # 9. AgentDojo's scoring rule
    utility = check_task_result(user_task, True, model_output or [], pre, post, traces)
    security = True
    if injection_task is not None:
        security = check_task_result(injection_task, False, model_output or [], pre, post, traces)

    n_tool_calls = sum(1 for m in messages if m["role"] == "tool")
    record = {
        "mode": "mcp", "suite": suite_name, "bench_version": bench, "model": model,
        "task": user_task_id, "injection_task": injection_task_id, "rep": rep,
        "utility": bool(utility), "security": bool(security),
        "llm_calls": client.llm_calls, "tok_in": client.tok_in, "tok_out": client.tok_out,
        "n_tool_calls": n_tool_calls, "wall_s": wall_s, "attempts": attempts,
        "model_output_none": model_output is None,
    }
    # X3 on a real episode: one trace record per tool-result message.
    if via_proxy:
        trace_records = len(verify_chain(ep_dir / "trace.jsonl"))   # raises if the chain is broken
        record["trace_records"] = trace_records
        record["trace_matches_tool_calls"] = trace_records == n_tool_calls

    # 10. result JSON (§3.4 format + extras), plus the full conversation
    (ep_dir / "messages.json").write_text(to_json(messages))
    (ep_dir / "result.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="banking")
    ap.add_argument("--bench", default="v1.2.2")
    ap.add_argument("--tasks", type=int, default=5, help="first N user tasks (numeric order)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--no-proxy", action="store_true", help="bridge -> server directly, no mcpfw")
    ap.add_argument("--fake-gt", action="store_true",
                    help="no key: fake model replays each task's ground truth")
    ap.add_argument("--force", action="store_true", help="replace existing episode folders")
    args = ap.parse_args()

    suite = get_suite(args.bench, args.suite)
    out_root = ROOT / "run" / ("fake" if args.fake_gt else "")
    if not args.fake_gt:
        real_client, model = make_client(fake=False)

    for rep in range(args.reps):
        for tid in numeric_task_ids(suite)[: args.tasks]:
            if args.fake_gt:
                task = suite.user_tasks[tid]
                env = task.init_environment(suite.load_and_inject_default_environment({}))
                calls = task.ground_truth(env)
                client, model = CountingClient(GroundTruthFakeClient(calls, task.GROUND_TRUTH_OUTPUT)), "fake-gt"
            else:
                client = CountingClient(real_client)
            r = run_episode(suite, args.suite, args.bench, tid, rep, client, model, out_root,
                            via_proxy=not args.no_proxy, force=args.force)
            extra = (f" trace={r['trace_records']} match={r['trace_matches_tool_calls']}"
                     if "trace_records" in r else "")
            print(f"{tid} r{rep}: utility={r['utility']} calls={r['llm_calls']} tok_in={r['tok_in']} "
                  f"tok_out={r['tok_out']} tools={r['n_tool_calls']} {r['wall_s']}s{extra}")


if __name__ == "__main__":
    main()
