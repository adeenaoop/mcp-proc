"""X2: wire transparency (guide §8.6). No LLM, no key.

Replays every banking user task's and injection task's ground_truth() calls,
plus the deliberate bad calls, through harness -> mcpfw (--tee) -> server.
Passes iff, for each run:
    sha256(up_in.bin)   == sha256(up_out.bin)     (harness -> server direction)
    sha256(down_in.bin) == sha256(down_out.bin)   (server -> harness direction)
Also checks the captures are non-empty, so an empty == empty match can't pass.

Run from ~/fyp:  pytest tests/test_wire.py -v
"""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_replay import BAD_CALLS, PROXY, SERVER, BENCH, SUITE, numeric, run_direct, suite  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from bridge import MCPBridge  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_through_proxy(calls, ep_dir, user_task_id=None, tee=True):
    server_args = [str(SERVER), "--suite", SUITE, "--bench-version", BENCH,
                   "--env-pre", str(ep_dir / "env_pre.json"),
                   "--env-out", str(ep_dir / "env_post.json")]
    if user_task_id:
        server_args += ["--user-task", user_task_id]
    proxy_args = [str(PROXY), "--log", str(ep_dir / "trace.jsonl")]
    if tee:
        proxy_args += ["--tee", str(ep_dir / "wire")]
    bridge = MCPBridge(sys.executable, proxy_args + ["--", sys.executable, *server_args],
                       expected_tools={f.name for f in suite.tools})
    try:
        out = [bridge.call(c.function, dict(c.args)) for c in calls]
    finally:
        bridge.close()
    return out


def check_wire(ep_dir):
    w = ep_dir / "wire"
    for name in ("up_in", "up_out", "down_in", "down_out"):
        assert (w / f"{name}.bin").stat().st_size > 0, f"{name}.bin is empty"
    assert sha(w / "up_in.bin") == sha(w / "up_out.bin"), "harness -> server bytes changed"
    assert sha(w / "down_in.bin") == sha(w / "down_out.bin"), "server -> harness bytes changed"


@pytest.mark.parametrize("task_id", numeric(suite.user_tasks))
def test_wire_user_task(task_id, tmp_path):
    calls = run_direct(suite.user_tasks[task_id], True)[0]
    run_through_proxy(calls, tmp_path, task_id)
    check_wire(tmp_path)


@pytest.mark.parametrize("task_id", numeric(suite.injection_tasks))
def test_wire_injection_task(task_id, tmp_path):
    calls = run_direct(suite.injection_tasks[task_id], False)[0]
    run_through_proxy(calls, tmp_path)
    check_wire(tmp_path)


def test_wire_error_calls(tmp_path):
    run_through_proxy(BAD_CALLS, tmp_path)
    check_wire(tmp_path)
